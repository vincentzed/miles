"""Collect the sampler's distribution at generation time for score centering."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.score_centering import score_centering_top_k, validate_score_centering_sampling
from miles.utils.types import Sample


def configure_score_centering_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    """Request candidate probabilities, validating the actual per-call settings."""
    k = score_centering_top_k(args)
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    if getattr(args, "sc_rollout_min_p", 0.0):
        sampling["min_p"] = args.sc_rollout_min_p
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        sampling.setdefault(key, default)
    filter_mode = getattr(args, "sc_filter_mode", "none")
    if filter_mode == "none":
        validate_score_centering_sampling(sampling, temperature=args.rollout_temperature)
    elif filter_mode != "pre":  # "pre" = naive: pre-filter candidates used as if they were q (research arm)
        if openai:
            raise ValueError("Filtered score centering is only wired for native /generate")
        # SGLang aborts sampling-mask requests without a finite top_k (scheduler.py:2787). Capping top_k at
        # the recorded head width also guarantees support S ⊆ head, so q^F is reconstructed exactly.
        sampling["top_k"] = k if sampling["top_k"] in (-1, 0) else min(sampling["top_k"], k)
        request["return_sampling_mask"] = True
    if openai:
        request["top_logprobs"] = k
    else:
        request["top_logprobs_num"] = max(k, request.get("top_logprobs_num", 0) or 0)


def append_score_centering_topk(sample: Sample, meta: Mapping[str, Any], k: int) -> None:
    """Append compact [response, k] arrays after appending generated tokens.

    No rescoring is allowed: with stale rollouts it would replace the behavior
    policy. Slots unavailable from the server are represented by ID -1/logp -inf.
    """
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    rows = meta.get("output_top_logprobs")
    if rows is None and n:
        raise ValueError("Score centering requires SGLang output_top_logprobs from generation")
    rows = rows if rows is not None else []
    if len(rows) != n:
        raise ValueError("Score-centering top-k rows do not match generated tokens")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows):
        if not entries:
            raise ValueError(f"Missing score-centering candidates at generated position {i}")
        entries = entries[:k]
        token_ids = [entry[1] for entry in entries]
        probabilities = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in token_ids):
            raise ValueError("Score-centering candidates must have non-negative integer token IDs")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Duplicate score-centering candidate token IDs")
        if np.isnan(probabilities).any() or (probabilities > 0).any() or np.exp(probabilities).sum() > 1 + 1e-5:
            raise ValueError("Invalid score-centering candidate probabilities")
        ids[i, : len(entries)] = token_ids
        logps[i, : len(entries)] = probabilities
    prefix_length = sample.response_length - n
    for field, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, field)
        if previous is None:
            if prefix_length:
                raise ValueError("Cannot start collecting score-centering candidates midway through a response")
            setattr(sample, field, values)
        else:
            if previous.shape != (prefix_length, k):
                raise ValueError(f"Misaligned {field}: {previous.shape}, expected {(prefix_length, k)}")
            setattr(sample, field, np.concatenate((previous, values)))


def apply_filtered_support(sample: Sample, meta: Mapping[str, Any], n: int) -> None:
    """Replace the last n rows of pre-filter candidates by the post-filter sampling distribution q^F.

    SGLang returns top-k logprobs and the sampled-token logprob *before* top-p/top-k/min-p filtering,
    plus (return_sampling_mask) the support S each token was actually drawn from. Every such filter keeps a
    prefix of the probability-sorted vocabulary, so S is the first |S| candidates whenever |S| <= k; then
    log q^F_v = log q_v - logsumexp_S log q for v in S and 0 mass elsewhere (exact). If |S| > k the head is
    truncated to the k best and renormalized over them (approximation; counted in sample.metadata).
    """
    masks = meta.get("output_token_sampling_mask")
    if masks is None or len(masks) != n:
        raise ValueError("Filtered score centering needs output_token_sampling_mask for every generated token")
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    start = sample.response_length - n
    truncated = 0
    for i, support in enumerate(masks):
        row = start + i
        head = {int(t): j for j, t in enumerate(ids[row]) if t >= 0}
        keep = [head[int(t)] for t in support if int(t) in head]
        truncated += len(keep) < len(support)
        lq = logps[row, keep].astype(np.float64)
        log_z = float(np.logaddexp.reduce(lq))
        new_ids = np.full(ids.shape[1], -1, dtype=np.int32)
        new_lp = np.full(ids.shape[1], -np.inf, dtype=np.float32)
        new_ids[: len(keep)] = ids[row, keep]
        new_lp[: len(keep)] = (lq - log_z).astype(np.float32)
        ids[row], logps[row] = new_ids, new_lp
        sample.rollout_log_probs[row] = float(sample.rollout_log_probs[row] - log_z)
    if truncated:
        raise ValueError(f"{truncated} positions have sampling support outside the recorded head (top_k must be <= k)")


def append_score_centering_observations(sample: Sample, count: int) -> None:
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_score_centering_sample(sample: Sample, k: int) -> None:
    """Validate the contract also for custom producers and restored rollouts."""
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError("Score centering requires candidate and sampled-token logprobs on every rollout")
    if ids.shape != (sample.response_length, k):
        raise ValueError("Score-centering candidates must have shape [response_length, score_centering_top_k]")
    if ids.dtype != np.int32 or logps.dtype != np.float32:
        raise ValueError("Score-centering candidates require int32 IDs and float32 logprobs")
    valid = ids >= 0
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (ids < -1).any() or (active & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs score-centering candidates")
    if (active & ~np.isfinite(logps).any(-1)).any():
        raise ValueError("Every trained token needs positive candidate probability mass")
    if np.isnan(logps).any() or (logps > 0).any() or (~np.isneginf(logps[~valid])).any():
        raise ValueError("Invalid score-centering logprobs or candidate padding")
    if (np.exp(logps.astype(np.float64)).sum(-1) > 1 + 1e-5).any():
        raise ValueError("Score-centering candidate probability mass exceeds one")
    # Batched sorting avoids one Python/NumPy call per generated token. Repeated
    # -1 padding is allowed; nonnegative candidate IDs must be unique per row.
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate score-centering candidate token IDs")
    sampled = np.asarray(sample.rollout_log_probs)
    if not np.isfinite(sampled[active]).all() or (sampled[active] > 0).any():
        raise ValueError("Invalid score-centering sampled-token logprobs")
    tokens = np.asarray(sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def merge_score_centering_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for score centering")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
