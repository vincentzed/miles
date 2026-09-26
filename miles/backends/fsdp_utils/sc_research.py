"""Research instrumentation for score-centering studies (arXiv:2609.20807).

Everything here is opt-in and inert unless the matching `--sc-*` flag is set:
- paper loss normalization: sum over tokens / total tokens of the optimizer batch;
- placebo advantages (constant or content-independent), for causal drift tests;
- drift decomposition: one extra backward of the centering term per optimizer step, giving the
  gradient split G_pg = G_sc - G_c and running sums / cross-step inner products;
- a jsonl sink shared with the weight-sync hook (weight-noise projection diagnostics).
"""

import json
import os
import zlib
from argparse import Namespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def append_diag(args: Namespace, record: dict) -> None:
    path = getattr(args, "sc_diag_path", None)
    if not path or dist.get_rank() != 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


def grads_of(model: torch.nn.Module) -> list[torch.Tensor]:
    return [_local(p.grad) if p.grad is not None else None for p in model.parameters()]


def scale_grads(model: torch.nn.Module, factor: float) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(factor)


def token_mean_grad_scale(count: torch.Tensor, dp_size: int) -> float:
    """Factor turning FSDP's rank-averaged gradient of summed token losses into the paper's token mean."""
    count = count.detach().float().clone()
    if dist.is_initialized():
        dist.all_reduce(count)
    return dp_size / max(count.item(), 1.0)


def log_advantage_stats(args: Namespace, rollout_data: dict, rollout_id: int) -> None:
    """Token-weighted mean advantage and Cov(A, L): with the batch token-mean loss the net drift weight is
    sum_i A_i L_i, so its sign predicts the direction of drift (toward / away from the sampler)."""
    if not getattr(args, "sc_diag_path", None):
        return
    adv = torch.tensor([float(a[0]) if len(a) else 0.0 for a in rollout_data["advantages"]], dtype=torch.float64)
    lens = torch.tensor([float(x) for x in rollout_data["response_lengths"]], dtype=torch.float64)
    record = {
        "kind": "adv",
        "rollout_id": rollout_id,
        "token_weighted_adv": float((adv * lens).sum() / lens.sum().clamp_min(1)),
        "cov_adv_len": float(((adv - adv.mean()) * (lens - lens.mean())).mean()),
        "mean_len": float(lens.mean()),
    }
    append_diag(args, record)


def apply_placebo_advantages(args: Namespace, rollout_data: dict, rollout_id: int) -> None:
    """Replace advantages by a content-independent placebo.

    plus_one / minus_one: A = +-1 for every token (uncentered). On-policy the expected update is
    zero (E_p[score] = 0), so any systematic motion is drift toward/away from the sampler.
    random_sign: one independent +-1 per sequence (seeded by rollout id); E[A | prefix] = 0.
    """
    shift = getattr(args, "sc_advantage_shift", 0.0)
    if shift:
        # Dose knob for drift: A + c keeps the signal (covariance) and adds c * s-bar of drift.
        rollout_data["advantages"] = [a + shift for a in rollout_data["advantages"]]
    mode = getattr(args, "sc_placebo_advantage", "none")
    if mode == "none":
        return
    advantages = rollout_data["advantages"]
    if mode == "random_sign":
        rank = dist.get_rank() if dist.is_initialized() else 0
        generator = torch.Generator().manual_seed((args.seed * 7919 + rollout_id) * 65537 + rank)
        signs = torch.randint(0, 2, (len(advantages),), generator=generator) * 2 - 1
    else:
        signs = torch.full((len(advantages),), 1 if mode == "plus_one" else -1)
    rollout_data["advantages"] = [torch.full_like(a, float(s)) for a, s in zip(advantages, signs.tolist())]


# Mini-float grids (mantissa bits, max, min normal exponent), as in the paper's models/quant.py.
_FLOAT_GRIDS = {"fp8": (3, 448.0, -6), "fp6": (2, 28.0, -2), "fp4": (1, 6.0, 0)}


def _snap_float(x: torch.Tensor, fmt: str) -> torch.Tensor:
    man_bits, max_val, min_normal_exp = _FLOAT_GRIDS[fmt]
    ax = x.abs()
    exp = torch.floor(torch.log2(ax.clamp_min(1e-30)))
    quantum = torch.exp2(exp.clamp_min(min_normal_exp) - man_bits)
    return torch.sign(x) * torch.minimum(torch.round(ax / quantum) * quantum, torch.tensor(max_val, device=x.device))


def fake_quantize_weight(name: str, w: torch.Tensor, fmt: str, group_size: int = 0) -> torch.Tensor:
    """Weight-only quantize-dequantize, port of the paper's fake_quantize_tree (models/quant.py).

    Symmetric absmax scale per output row over the contraction (last) axis, optionally in groups.
    Skips 1-D tensors, norms, biases and MoE routers. fmt: int8 | int4 | intN | fp8 | fp6 | fp4.
    """
    if w.ndim <= 1 or "norm" in name or "bias" in name or name.endswith("mlp.gate.weight"):
        return w
    shape = w.shape
    k = shape[-1]
    g = group_size or k
    assert k % g == 0, f"{name}: contraction {k} not divisible by group {g}"
    x = w.float().reshape(*shape[:-1], k // g, g)
    if fmt.startswith("int"):
        qmax = float(2 ** (int(fmt[3:]) - 1) - 1)
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / qmax
        q = torch.round(x / scale).clamp(-qmax, qmax)
    else:
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / _FLOAT_GRIDS[fmt][1]
        q = _snap_float(x / scale, fmt)
    return (q * scale).reshape(shape).to(w.dtype)


class SamplerView:
    """What the rollout engine receives on each weight sync (research knobs, all opt-in).

    mode "normal": on a refresh step send theta_t + Delta (Delta = sigma * eps * theta_0, drawn once
    from the initial sync, seeded by --sc-weight-noise-seed) and snapshot it; between refreshes
    (--sc-stale-interval N: refresh every N syncs, the paper's staleness s gives N = s + 1) resend the
    snapshot, so the sampler is exactly N-stale as in the paper (colocated SGLang drops its weights on
    offload, so skipping the sync is not an option). mode "clean": send theta_t (clean eval).
    mode "restore": resend the snapshot after a clean eval.
    """

    def __init__(self, args: Namespace, model: torch.nn.Module):
        self.args = args
        self.sigma = getattr(args, "sc_weight_noise", 0.0)
        self.interval = max(1, getattr(args, "sc_stale_interval", 1))
        self.keep_snapshot = self.interval > 1 or getattr(args, "sc_clean_eval", False)
        self.fake_quant = getattr(args, "sc_sampler_weight_quant", None)
        self.fake_quant_group = getattr(args, "sc_sampler_weight_quant_group", 0)
        self.enabled = bool(self.sigma) or self.keep_snapshot or bool(self.fake_quant)
        tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        self._alias = {"lm_head.weight": "model.embed_tokens.weight"} if tied else {}
        self.deltas: dict[str, torch.Tensor] = {}
        self.theta0: dict[str, torch.Tensor] = {}
        self._drawn_at: dict[str, int] = {}
        self.n_sham = 8  # independent never-applied directions: null distribution for the Delta-projection
        self.snapshot: dict[str, torch.Tensor] = {}
        self.normal_syncs = 0
        self.mode = "normal"
        self.refresh = True
        self._stats: dict[str, float] = {}

    def begin(self, mode: str) -> None:
        self.mode = mode
        if mode == "normal":
            self.normal_syncs += 1
            self.refresh = (self.normal_syncs - 1) % self.interval == 0
        self._stats = {"proj": 0.0, "disp_sq": 0.0, "delta_sq": 0.0}

    def transform(self, name: str, full: torch.Tensor, target_dtype: torch.dtype | None) -> torch.Tensor:
        sent_dtype = target_dtype or full.dtype
        if self.mode == "clean":
            return full.to(sent_dtype)
        if self.mode == "restore" or not self.refresh:
            return self.snapshot[name].to(full.device, non_blocking=True)
        out = full.to(sent_dtype)
        if self.sigma:
            key = self._alias.get(name, name)
            # --sc-noise-redraw: every refresh draws a fresh Delta (breaks the fixed-bias feedback loop)
            if getattr(self.args, "sc_noise_redraw", False) and self._drawn_at.get(key) != self.normal_syncs:
                self.deltas.pop(key, None)
            if key not in self.deltas:
                redraw = self.normal_syncs if getattr(self.args, "sc_noise_redraw", False) else 0
                seed = (self.args.sc_weight_noise_seed * 1_000_003 + zlib.crc32(key.encode()) + redraw * 7_919_111) % 2**63
                self._drawn_at[key] = self.normal_syncs
                generator = torch.Generator(device=full.device).manual_seed(seed)
                eps = torch.randn(full.shape, generator=generator, device=full.device, dtype=torch.float32)
                self.deltas[key] = (self.sigma * eps * full.float()).to(sent_dtype)
                self.theta0.setdefault(key, full.detach().float().clone())
            if key == name:  # count tied weights once
                delta = self.deltas[key].float()
                disp = full.float() - self.theta0[key]
                self._stats["proj"] += (disp * delta).sum().item()
                self._stats["disp_sq"] += disp.square().sum().item()
                self._stats["delta_sq"] += delta.square().sum().item()
                base = self.theta0[key] * self.sigma
                for j in range(self.n_sham):
                    g = torch.Generator(device=full.device).manual_seed((zlib.crc32(key.encode()) * 131 + j * 7919 + 17) % 2**63)
                    sham = torch.randn(full.shape, generator=g, device=full.device, dtype=torch.float32) * base
                    self._stats[f"sham_proj{j}"] = self._stats.get(f"sham_proj{j}", 0.0) + (disp * sham).sum().item()
                    self._stats[f"sham_sq{j}"] = self._stats.get(f"sham_sq{j}", 0.0) + sham.square().sum().item()
            out = out + self.deltas[key]  # paper: bf16 weight + bf16 delta
        if self.fake_quant:
            # Re-applied on every sync, so the error tracks the current weights (paper: noise, then quant).
            out = fake_quantize_weight(name, out, self.fake_quant, self.fake_quant_group)
        if self.keep_snapshot:
            self.snapshot[name] = out.detach().to("cpu", copy=True)
        return out

    def end(self, weight_version: int) -> None:
        record = {"kind": "sync", "mode": self.mode, "weight_version": weight_version, "refresh": self.refresh}
        if self.sigma and self.mode == "normal" and self.refresh:
            s = self._stats
            record.update(
                delta_proj=s["proj"] / max(s["delta_sq"], 1e-30),
                delta_cos=s["proj"] / max((s["disp_sq"] * s["delta_sq"]) ** 0.5, 1e-30),
                disp_norm=s["disp_sq"] ** 0.5,
                delta_norm=s["delta_sq"] ** 0.5,
                sham_proj=[s.get(f"sham_proj{j}", 0.0) / max(s.get(f"sham_sq{j}", 0.0), 1e-30) for j in range(self.n_sham)],
            )
        append_diag(self.args, record)


class DriftTracker:
    """Split each step's gradient into the centering component G_c and the rest.

    With the score-centering loss L = -A (w log p - c), G_total = G_pg + G_c where G_c = grad(A c).
    For a PG arm (centering disabled) the applied gradient is G_pg; for an SC arm it is G_pg + G_c.
    Under SGD, theta_T - theta_0 = -lr * sum_t G_t exactly, so the running sums of G_c and G_pg
    give the accumulated drift vs signal displacement. Cross-step inner products use different
    batches but are NOT unbiased: step t's weights depend on batch t-1, so E<G_t, G_{t-1}> carries an
    O(lr * tr(H Sigma)) term. Treat them as descriptive; the sum-norm growth rate is the main statistic.
    """

    def __init__(self, model: torch.nn.Module):
        params = [_local(p) for p in model.parameters()]
        self.sum_c = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        self.sum_pg = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        self.prev_c = None
        self.prev_pg = None
        self.steps = 0

    @staticmethod
    def _dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
        total = torch.zeros((), device=a[0].device, dtype=torch.float64)
        for x, y in zip(a, b, strict=True):
            total += (x.double() * y.double()).sum()
        if dist.is_initialized():
            dist.all_reduce(total)
        return total.item()

    def split_half(self, g_total, half_total, g_c, half_c, grad_scale, center_scale) -> dict:
        """Cross-half products (x4 to undo the halving): approx. unbiased for ||E G_c||^2, ||E G_pg||^2 and
        <E G_c, E G_pg>. Halves are microbatch-contiguous, not group-aligned, so group-centered advantages
        leave a small cross-half correlation; treat as approximate."""
        if half_total is None or half_c is None:
            return {}
        ca = [h.float() * grad_scale for h in half_c]
        cb = [c.float() - a for c, a in zip(g_c, ca, strict=True)]
        ta = [h.float() * grad_scale for h in half_total]
        tb = [t.float() - a for t, a in zip(g_total, ta, strict=True)]
        pa = [t - center_scale * c for t, c in zip(ta, ca, strict=True)]
        pb = [t - center_scale * c for t, c in zip(tb, cb, strict=True)]
        return {
            "split_c": 4 * self._dot(ca, cb),
            "split_pg": 4 * self._dot(pa, pb),
            "split_cross": 2 * (self._dot(ca, pb) + self._dot(pa, cb)),
        }

    def observe(self, g_total: list[torch.Tensor], g_c: list[torch.Tensor], center_scale: float) -> dict:
        """center_scale = lambda actually applied (0 for PG arms): G_total = G_pg + lambda * G_c."""
        g_c = [g.float().clone() for g in g_c]
        g_pg = [t.float() - center_scale * c for t, c in zip(g_total, g_c, strict=True)]
        record = {
            "g_c_sq": self._dot(g_c, g_c),
            "g_pg_sq": self._dot(g_pg, g_pg),
            "g_c_dot_g_pg": self._dot(g_c, g_pg),
        }
        if self.prev_c is not None:
            record["g_c_cross_step"] = self._dot(g_c, self.prev_c)
            record["g_pg_cross_step"] = self._dot(g_pg, self.prev_pg)
        for s, g in zip(self.sum_c, g_c, strict=True):
            s.add_(g)
        for s, g in zip(self.sum_pg, g_pg, strict=True):
            s.add_(g)
        self.steps += 1
        record["sum_c_sq"] = self._dot(self.sum_c, self.sum_c)
        record["sum_pg_sq"] = self._dot(self.sum_pg, self.sum_pg)
        record["sum_c_dot_sum_pg"] = self._dot(self.sum_c, self.sum_pg)
        self.prev_c, self.prev_pg = g_c, g_pg
        return record
