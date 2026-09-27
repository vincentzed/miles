"""Emulated NVFP4 backward GEMMs for the FSDP actor (research: can RL gradients run through FP4?).

For y = x W^T the backward has two GEMMs:
  dgrad  dx = dy W        reduction over the output dim  -> quantize dy rows (1x16 along out) and W along out
  wgrad  dW = dy^T x      reduction over TOKENS          -> quantize dy and x along the token axis (16-token blocks)
Each operand is NVFP4 quantize-dequantized (E2M1, 16-element blocks, E4M3 block scales, FP32 tensor amax; same math
as TE's NVFP4Quantizer), then multiplied in BF16 with FP32 accumulation, which is what an FP4 GEMM computes up to
accumulation order.

Options (comma list, e.g. "rtn", "sr", "rht,sr", "sort", "sort,balance"):
  sr       stochastic rounding of the gradient operand dy (NVIDIA pretraining recipe: unbiased gradients)
  rht      16x16 random-sign Hadamard along the token axis on both wgrad operands (exact: H^T H = I)
  sort     permute tokens by ||dy_t|| before wgrad quantization (exact: the sum over tokens is order-free) so each
           16-token block holds similar-magnitude tokens; RL gradients are concentrated in few tokens, so unsorted
           blocks round the small ones to zero
  balance  rescale token t by c_t = sqrt(||x_t|| / ||dy_t||) on dy and 1/c_t on x before wgrad quantization (exact)
  wgrad_bf16 / dgrad_bf16   keep that GEMM in BF16 (ablations)
  dgrad_fp8  dgrad GEMM operands in FP8 E4M3 with per-32-element block scales (MXFP8-like) instead of FP4
  layers=a-b  only decoder layers a..b (inclusive) use the emulated backward; the rest keep BF16 backward
  wfwd     dgrad reuses the forward weight as is (it must already be on a 16x16 grid, e.g. QAT with nvfp4_2d), which is
           what TE's 2D weight quantization gives: one quantization serves fprop and dgrad (chain-rule consistent)
"""

import math

import torch

from miles.backends.fsdp_utils.sc_research import fake_quantize_weight

_H16: torch.Tensor | None = None


def _hadamard16(device) -> torch.Tensor:
    global _H16
    if _H16 is None or _H16.device != device:
        h = torch.tensor([[1.0]])
        for _ in range(4):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H16 = (h / 4.0).to(device)
    return _H16


def _qdq_rows(t: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """NVFP4 QDQ along the last dim (1x16 blocks). Pads the last dim to a multiple of 16 with zeros."""
    k = t.shape[-1]
    pad = (-k) % 16
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    q = fake_quantize_weight("fp4bwd", t.to(torch.bfloat16).contiguous(), "nvfp4", 16, generator=generator)
    return q[..., :k] if pad else q


def _qdq_rows_fp8(t: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 QDQ with per-32-element block absmax scales along the last dim."""
    k = t.shape[-1]
    pad = (-k) % 32
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    q = fake_quantize_weight("fp8bwd", t.to(torch.bfloat16).contiguous(), "fp8", 32)
    return q[..., :k] if pad else q


class FP4BackwardConfig:
    def __init__(self, spec: str, seed: int = 0):
        self.opts = {o.strip() for o in spec.split(",") if o.strip() and not o.strip().startswith("layers=")}
        rng = [o.strip()[7:] for o in spec.split(",") if o.strip().startswith("layers=")]
        self.layers = tuple(int(v) for v in rng[0].split("-")) if rng else None
        unknown = self.opts - {"rtn", "sr", "rht", "sort", "balance", "wgrad_bf16", "dgrad_bf16", "wfwd", "dgrad_fp8"}
        assert not unknown, f"unknown fp4 backward options {unknown}"
        self.seed = seed
        self.calls = 0

    def gen(self, device):
        if "sr" not in self.opts:
            return None
        self.calls += 1
        return torch.Generator(device=device).manual_seed(self.seed * 1_000_003 + self.calls)


class _FP4BwdLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cfg):
        ctx.save_for_backward(x, w)
        ctx.cfg = cfg
        return torch.nn.functional.linear(x, w)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        cfg: FP4BackwardConfig = ctx.cfg
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        dx = dw = None
        if ctx.needs_input_grad[0]:
            if "dgrad_bf16" in cfg.opts:
                dx = dy2 @ w.to(dy2.dtype)
            else:  # dy rows along out-dim; W along out-dim = rows of W^T
                if "dgrad_fp8" in cfg.opts:
                    dyq = _qdq_rows_fp8(dy2)
                    wq = _qdq_rows_fp8(w.t().contiguous()).t()
                else:
                    dyq = _qdq_rows(dy2, cfg.gen(dy2.device))
                    wq = w if "wfwd" in cfg.opts else _qdq_rows(w.t().contiguous()).t()
                dx = (dyq.float() @ wq.float()).to(dy2.dtype)
            dx = dx.reshape(shape)
        if ctx.needs_input_grad[1]:
            if "wgrad_bf16" in cfg.opts:
                dw = (dy2.float().t() @ x2.float()).to(w.dtype)
            else:
                a, b = dy2.float(), x2.float()  # tokens along dim 0
                n = a.shape[0]
                if "balance" in cfg.opts:
                    c = (b.norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)).clamp(1e-6, 1e6).sqrt()
                    c = torch.where(a.norm(dim=1) > 0, c, torch.ones_like(c))
                    a, b = a * c[:, None], b / c[:, None]
                if "sort" in cfg.opts:
                    perm = torch.argsort(a.norm(dim=1), descending=True)
                    a, b = a[perm], b[perm]
                pad = (-n) % 16
                if pad:
                    a = torch.nn.functional.pad(a, (0, 0, 0, pad))
                    b = torch.nn.functional.pad(b, (0, 0, 0, pad))
                if "rht" in cfg.opts:  # same random-sign 16x16 Hadamard on each 16-token block of both operands
                    h = _hadamard16(a.device)
                    g = torch.Generator(device=a.device).manual_seed(cfg.seed * 7_919 + cfg.calls)
                    s = (torch.randint(0, 2, (16,), generator=g, device=a.device) * 2 - 1).float()
                    hs = h * s[None, :]
                    a = (hs @ a.reshape(-1, 16, a.shape[1])).reshape(a.shape)
                    b = (hs @ b.reshape(-1, 16, b.shape[1])).reshape(b.shape)
                aq = _qdq_rows(a.t().contiguous(), cfg.gen(a.device))  # [out, tokens], blocks along tokens
                bq = _qdq_rows(b.t().contiguous())  # [in, tokens]
                dw = (aq.float() @ bq.float().t()).to(w.dtype)
        return dx, dw, None


def install(model: torch.nn.Module, spec: str, seed: int = 0, include=("q_proj", "k_proj", "v_proj", "o_proj",
                                                                      "gate_proj", "up_proj", "down_proj")) -> int:
    """Route every decoder-block linear (no bias) through the emulated FP4 backward. Returns the count."""
    cfg = FP4BackwardConfig(spec, seed)
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.bias is None and name.split(".")[-1] in include and ".layers." in f".{name}":
            if cfg.layers is not None:
                idx = int(name.split(".layers.")[1].split(".")[0])
                if not cfg.layers[0] <= idx <= cfg.layers[1]:
                    continue
            def fwd(x, _m=mod):
                return _FP4BwdLinear.apply(x, _m.weight, cfg)
            mod.forward = fwd
            count += 1
    return count
