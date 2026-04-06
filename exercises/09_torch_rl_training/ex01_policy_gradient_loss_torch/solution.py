"""
Solution 01: PPO Clipped Policy Gradient Loss in PyTorch
"""

import torch


def compute_ppo_clipped_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    eps_clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO clipped policy gradient loss.

    Args:
        old_log_probs: (batch, seq_len) old policy log-probs, detached.
        new_log_probs: (batch, seq_len) current policy log-probs, requires grad.
        advantages:    (batch, seq_len) advantage values.
        loss_mask:     (batch, seq_len) binary mask for valid tokens.
        eps_clip:      PPO clipping epsilon.

    Returns:
        loss: scalar, masked mean of per-token clipped loss.
        clip_frac: scalar, fraction of valid tokens that were clipped.
    """
    # Step 1: Compute importance sampling ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Step 2: Unclipped surrogate loss (negative because we minimize)
    surr1 = -ratio * advantages

    # Step 3: Clipped surrogate loss
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    surr2 = -clipped_ratio * advantages

    # Step 4: Pessimistic (max) clipping -- take the worse (larger) loss
    per_token_loss = torch.maximum(surr1, surr2)

    # Step 5: Masked mean
    num_valid = torch.clamp_min(loss_mask.sum(), 1.0)
    loss = (per_token_loss * loss_mask).sum() / num_valid

    # Step 6: Clip fraction (non-differentiable metric)
    clip_frac = ((surr2 > surr1).float() * loss_mask).sum() / num_valid

    return loss, clip_frac
