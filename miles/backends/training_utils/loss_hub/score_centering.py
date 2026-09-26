"""Score centering from arXiv:2609.20807, Appendix A.

The sampler distribution is fixed data. Both the importance weights and the
head residual must be detached: differentiating either changes the estimator.
"""

import math
from typing import Any

import torch
import torch.distributed as dist


def importance_weights(
    log_ratio: torch.Tensor,
    mode: str,
    *,
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
    positive: torch.Tensor | None = None,
    ppo_low: float = 0.8,
    ppo_high: float = 1.2,
) -> torch.Tensor:
    """Evaluate token-level weights without exponentiating unbounded ratios."""
    if mode == "none":
        return torch.ones_like(log_ratio)
    if mode == "tis":
        return log_ratio.clamp(max=math.log(tis_clip)).exp()
    if mode == "mis":
        inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
        return torch.where(inside, log_ratio.clamp(max=math.log(mis_high)).exp(), 0.0)
    if mode == "ppo":
        inside = ppo_in_band(log_ratio, positive, low=ppo_low, high=ppo_high)
        return torch.where(inside, log_ratio.clamp(max=math.log(max(ppo_high, PPO_DUAL_CLIP))).exp(), 0.0)
    raise ValueError(f"Unknown score-centering importance weighting: {mode}")


PPO_DUAL_CLIP = 3.0


def ppo_in_band(log_ratio: torch.Tensor, positive: torch.Tensor, *, low: float, high: float) -> torch.Tensor:
    """PPO's pessimistic clip as a detached REINFORCE weight (paper code, losses.py:151-160).

    Gradient-identical to the clipped surrogate: A >= 0 keeps r <= high, A < 0 keeps
    low <= r <= 3 (dual clip); outside the band the token is dropped. The band depends on
    sign(A), so centering is done conditionally on the sign (losses.py:192-196).
    """
    if positive.dim() < log_ratio.dim():
        positive = positive.unsqueeze(-1)
    upper = torch.where(positive, math.log(high), math.log(PPO_DUAL_CLIP))
    lower = torch.where(positive, -math.inf, math.log(low))
    return (log_ratio <= upper) & (log_ratio >= lower)


def score_centering_loss(
    train_log_probs: torch.Tensor,
    train_head_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    rollout_head_log_probs: torch.Tensor,
    head_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    mode: str = "none",
    tis_clip: float = 2.0,
    mis_low: float = 0.5,
    mis_high: float = 5.0,
    eps: float = 1e-6,
    center: bool = True,
    center_only: bool = False,
    ppo_low: float = 0.8,
    ppo_high: float = 1.2,
    q_tail_floor: float | None = None,
    center_scale: float = 1.0,
    old_log_probs: torch.Tensor | None = None,
    old_head_log_probs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return unreduced token losses and detached token metrics.

    Inputs have shape [tokens], except head tensors/mask [tokens, candidates].
    Missing head slots have a false mask. Tail masses use the appendix's floor;
    cancellation is exact for a full distribution, approximate for a true tail
    that differs from the modeled, rescaled trainer tail.
    """
    head_log_probs = torch.where(head_mask, train_head_log_probs, 0.0)
    with torch.no_grad():
        p = head_log_probs.exp().masked_fill(~head_mask, 0.0)
        q = rollout_head_log_probs.exp().masked_fill(~head_mask, 0.0)
        p_mass, q_mass = p.sum(-1), q.sum(-1)
        # The paper's appendix snippet floors the sampler tail at eps; its training code floors at 0.
        q_floor = eps if q_tail_floor is None else q_tail_floor
        rho = (1 - q_mass).clamp_min(q_floor) / (1 - p_mass).clamp_min(eps)
        positive = advantages.detach() >= 0
        if mode == "none":
            weighted_q, alpha = q, rho
        elif mode == "tis":
            # q * min(p/q, c), including q=0, without 0 * inf.
            weighted_q, alpha = torch.minimum(p, tis_clip * q), (tis_clip * rho).clamp_max(1)
        elif mode == "mis":
            log_ratio = head_log_probs - rollout_head_log_probs
            inside = (log_ratio >= math.log(mis_low)) & (log_ratio <= math.log(mis_high))
            weighted_q = torch.where(inside & head_mask, p, 0.0)
            alpha = ((rho >= 1 / mis_high) & (rho <= 1 / mis_low)).to(p.dtype)
        elif mode == "ppo_old":
            # PPO clipped against pi_old (trainer at batch start), rollouts from the sampler q. Weight
            # w = r_old 1{band(r_old, sign A)}, r_old = p/pi_old; centering integrates q * w under q:
            # head q_v p_v/pi_old_v in band; tail q = rho p, pi_old = rho_o p -> alpha = (rho/rho_o) 1{band(1/rho_o)}.
            old_head = torch.where(head_mask, old_head_log_probs, 0.0)
            log_r = head_log_probs - old_head
            inside = ppo_in_band(log_r, positive, low=ppo_low, high=ppo_high)
            weighted_q = torch.where(inside & head_mask, q * log_r.exp(), 0.0)
            old_mass = old_head.exp().masked_fill(~head_mask, 0.0).sum(-1)
            rho_o = (1 - old_mass).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
            tail_in = ppo_in_band(-rho_o.log(), positive, low=ppo_low, high=ppo_high).to(p.dtype)
            alpha = rho / rho_o * tail_in
        elif mode == "ppo":
            # In band the effective mass q * (p/q) is p; out of band it is 0. The modeled tail has
            # constant ratio 1/rho, so alpha = rho * w(1/rho) = 1{1/rho in band}.
            inside = ppo_in_band(head_log_probs - rollout_head_log_probs, positive, low=ppo_low, high=ppo_high)
            weighted_q = torch.where(inside & head_mask, p, 0.0)
            tail_log_ratio = -rho.clamp_min(1e-30).log()
            alpha = ppo_in_band(tail_log_ratio, positive, low=ppo_low, high=ppo_high).to(p.dtype)
        else:
            raise ValueError(f"Unknown score-centering importance weighting: {mode}")
        residual = weighted_q - alpha.unsqueeze(-1) * p
        weight = importance_weights(
            train_log_probs - (old_log_probs if mode == "ppo_old" else rollout_log_probs),
            "ppo" if mode == "ppo_old" else mode,
            tis_clip=tis_clip,
            mis_low=mis_low,
            mis_high=mis_high,
            positive=positive,
            ppo_low=ppo_low,
            ppo_high=ppo_high,
        )
    correction =(residual * head_log_probs).sum(-1)
    # center_scale (lambda) is a causal control: 0 = PG, 1 = SC, 2 = same-size drift of opposite sign.
    applied_correction = correction * center_scale if center else correction * 0.0
    loss = -advantages.detach() * (weight * train_log_probs - applied_correction)
    if center_only:
        # Research diagnostic: gradient of the centering term alone (G_c), independent of `center`.
        loss = advantages.detach() * correction
    return loss, {
        "sc_correction": applied_correction.detach(),
        "sc_uncentered_correction": correction.detach(),
        "sc_importance_weight_sq": weight.square(),
        "sc_train_head_mass": p_mass,
        "sc_rollout_head_mass": q_mass,
        "sc_tail_ratio": rho,
        # sampler mass in its own top-128 (comparable across k; for k > 128 the head is sorted by q)
        "sc_rollout_head_mass_top128": q[..., :128].sum(-1),
        # ||q_hat - p||_1 of the modeled sampler distribution (head exact, tail rho*p): the logit-space
        # size of the centering vector, i.e. the per-token drift magnitude.
        "sc_head_l1": (q - p).abs().sum(-1) + (rho - 1).abs() * (1 - p_mass).clamp_min(0),
        "sc_importance_weight": weight,
    }


class _SelectedLogProbs(torch.autograd.Function):
    """Vocabulary-sharded log-softmax gather with a replicated loss on TP ranks.

    Only selected logits and normalization scalars are communicated. Backward
    computes each rank's vocabulary slice directly; reducing identical output
    gradients across TP ranks would incorrectly multiply the gradient by TP.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        group: dist.ProcessGroup | None,
        vocab_size: int,
        with_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank(group) if group is not None else 0
        width = logits.shape[-1]
        valid = token_ids >= 0
        local_ids = token_ids - rank * width
        local = valid & (local_ids >= 0) & (local_ids < width)
        local_ids = local_ids.clamp(0, width - 1)
        padded = torch.arange(width, device=logits.device) + rank * width >= vocab_size
        work = logits.masked_fill(padded, -torch.inf)
        maximum = work.amax(-1, keepdim=True)
        if group is not None:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        probabilities = (work - maximum).exp()
        denominator = probabilities.sum(-1, keepdim=True)
        selected = torch.where(local, work.gather(-1, local_ids), 0.0)
        if group is not None:
            dist.all_reduce(denominator, group=group)
            dist.all_reduce(selected, group=group)
        probabilities.div_(denominator)
        entropy = probabilities.new_zeros(probabilities.size(0))
        if with_entropy:
            entropy = -(probabilities * torch.where(probabilities > 0, probabilities.log(), 0.0)).sum(-1)
            if group is not None:
                dist.all_reduce(entropy, group=group)
        ctx.with_entropy = with_entropy
        ctx.save_for_backward(probabilities, local_ids, local, valid, entropy)
        return torch.where(valid, selected - maximum - denominator.log(), 0.0), entropy

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_entropy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        probabilities, local_ids, local, valid, entropy = ctx.saved_tensors
        grad = grad_output.masked_fill(~valid, 0.0)
        grad_logits = -probabilities * grad.sum(-1, keepdim=True)
        grad_logits.scatter_add_(-1, local_ids, grad.masked_fill(~local, 0.0))
        if ctx.with_entropy:
            logp = torch.where(probabilities > 0, probabilities.log(), 0.0)
            grad_logits -= grad_entropy.unsqueeze(-1) * probabilities * (logp + entropy.unsqueeze(-1))
        return grad_logits, None, None, None, None


def selected_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
) -> torch.Tensor:
    """Log-probabilities at global IDs [T, K] from local logits [T, V/TP].

    -1 is a padding ID and produces zero with zero gradient. The true vocabulary
    size excludes Megatron's padded vocabulary entries from normalization.
    """
    return selected_log_probs_and_entropy(
        logits, token_ids, group=group, vocab_size=vocab_size, temperature=temperature, chunk_size=chunk_size
    )[0]


def selected_log_probs_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    vocab_size: int | None = None,
    temperature: float = 1.0,
    chunk_size: int = -1,
    with_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected logprobs and optional entropy of the same unpadded distribution."""
    size = dist.get_world_size(group) if group is not None else 1
    vocab_size = vocab_size or logits.size(-1) * size
    if temperature <= 0 or not 0 < vocab_size <= logits.size(-1) * size:
        raise ValueError("Score centering needs a positive temperature and valid vocabulary size")
    if token_ids.ndim != 2 or logits.ndim != 2 or logits.size(0) != token_ids.size(0):
        raise ValueError("Expected logits [T, V/TP] and selected token IDs [T, K]")
    if ((token_ids < -1) | (token_ids >= vocab_size)).any():
        raise ValueError("Score-centering token ID is outside the model vocabulary")
    if logits.size(0) == 0:
        return logits.sum(-1, keepdim=True).expand_as(token_ids), logits.sum(-1)
    chunk_size = chunk_size if chunk_size > 0 else logits.size(0)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    chunks = [
        _SelectedLogProbs.apply(chunk.to(dtype) / temperature, ids, group, vocab_size, with_entropy)
        for chunk, ids in zip(logits.split(chunk_size), token_ids.split(chunk_size), strict=True)
    ]
    return torch.cat([chunk[0] for chunk in chunks]), torch.cat([chunk[1] for chunk in chunks])
