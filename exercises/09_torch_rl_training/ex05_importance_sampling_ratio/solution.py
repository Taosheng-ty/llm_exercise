"""
Solution 05: Importance Sampling Ratios
"""

import torch


def compute_token_is_ratio(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token importance sampling ratio.

    Args:
        new_log_probs: (batch, seq_len) current policy log-probs.
        old_log_probs: (batch, seq_len) rollout policy log-probs.

    Returns:
        ratio: (batch, seq_len) per-token IS ratios.
    """
    return torch.exp(new_log_probs - old_log_probs)


def compute_sequence_is_ratio(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence importance sampling ratio.

    Args:
        new_log_probs: (batch, seq_len) current policy log-probs.
        old_log_probs: (batch, seq_len) rollout policy log-probs.
        loss_mask:     (batch, seq_len) binary mask.

    Returns:
        seq_ratio: (batch,) per-sequence IS ratios.
    """
    log_ratio = (new_log_probs - old_log_probs) * loss_mask
    seq_log_ratio = log_ratio.sum(dim=-1)
    return torch.exp(seq_log_ratio)


def truncated_importance_sampling(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute truncated importance sampling weights.

    Args:
        new_log_probs: (batch, seq_len) current policy log-probs.
        old_log_probs: (batch, seq_len) rollout policy log-probs.
        loss_mask:     (batch, seq_len) binary mask.
        clip_low:      lower clip bound for ratio.
        clip_high:     upper clip bound for ratio.

    Returns:
        tis_weights:  (batch, seq_len) clipped per-token IS ratios.
        clip_frac:    scalar, fraction of valid tokens that were clipped.
        raw_ratio:    (batch, seq_len) unclipped ratios (for logging).
    """
    # Step 1: Compute raw per-token IS ratios
    raw_ratio = torch.exp(new_log_probs - old_log_probs)

    # Step 2: Clip for stability
    tis_weights = torch.clamp(raw_ratio, min=clip_low, max=clip_high)

    # Step 3: Clip fraction -- fraction of valid tokens where clipping occurred
    was_clipped = (tis_weights != raw_ratio).float()
    num_valid = torch.clamp_min(loss_mask.sum(), 1.0)
    clip_frac = (was_clipped * loss_mask).sum() / num_valid

    return tis_weights, clip_frac, raw_ratio
