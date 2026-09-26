"""Research instrumentation for score-centering studies (arXiv:2609.20807).

Everything here is opt-in and inert unless the matching `--sc-*` flag is set:
- paper loss normalization: sum over tokens / total tokens of the optimizer batch;
- placebo advantages (constant or content-independent), for causal drift tests;
- a jsonl sink for research diagnostics.
"""

import json
import os
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
    # within-group corr(R, sequence log q) (Power-distribution paper, Prop. 3) and a repetition readout
    seq_logq = torch.tensor([float(sum(x)) for x in rollout_data["rollout_log_probs"]], dtype=torch.float64)
    g = 8
    corrs = []
    for i in range(0, len(adv) - g + 1, g):
        a, lq = adv[i : i + g], seq_logq[i : i + g]
        if a.std() > 0 and lq.std() > 0:
            corrs.append(float(torch.corrcoef(torch.stack([a, lq]))[0, 1]))
    rep = []
    for toks, L in zip(rollout_data["tokens"], rollout_data["response_lengths"]):
        resp = [int(t) for t in toks[-int(L):]] if int(L) else []
        grams = [tuple(resp[j : j + 4]) for j in range(max(len(resp) - 3, 0))]
        rep.append(1 - len(set(grams)) / len(grams) if grams else 0.0)
    record = {
        "kind": "adv",
        "within_group_corr_adv_logq": float(sum(corrs) / len(corrs)) if corrs else None,
        "repeat_4gram_frac": float(sum(rep) / max(len(rep), 1)),
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
    if getattr(args, "sc_zero_token_mean_adv", False):
        # Remove the drift coefficient of the token-mean loss: subtract sum_i A_i L_i / sum_i L_i from every
        # token's advantage (per optimizer batch), keeping the within-batch covariance (the signal) intact.
        lens = [float(len(a)) for a in rollout_data["advantages"]]
        num = sum(float(a[0]) * l for a, l in zip(rollout_data["advantages"], lens) if len(a))
        c = num / max(sum(lens), 1.0)
        rollout_data["advantages"] = [a - c for a in rollout_data["advantages"]]
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

