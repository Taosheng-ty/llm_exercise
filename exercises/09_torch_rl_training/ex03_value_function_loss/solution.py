"""
Solution 03: Clipped Value Function Loss
"""

import torch


def compute_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: torch.Tensor,
    value_clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute clipped value function loss.

    Args:
        values:     (batch, seq_len) current value predictions, requires grad.
        old_values: (batch, seq_len) old value predictions, detached.
        returns:    (batch, seq_len) GAE returns.
        loss_mask:  (batch, seq_len) binary mask.
        value_clip: clipping threshold.

    Returns:
        loss: scalar, masked mean of per-token clipped value loss.
        clip_frac: scalar, fraction of valid tokens where |values - old_values| > value_clip.
    """
    # Step 1: Clip the value predictions
    values_clipped = old_values + torch.clamp(
        values - old_values, -value_clip, value_clip
    )

    # Step 2: Compute squared errors
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2

    # Step 3: Pessimistic (max) loss
    per_token_loss = torch.maximum(surr1, surr2)

    # Step 4: Masked mean
    num_valid = torch.clamp_min(loss_mask.sum(), 1.0)
    loss = (per_token_loss * loss_mask).sum() / num_valid

    # Step 5: Clip fraction
    clip_frac = ((torch.abs(values - old_values) > value_clip).float() * loss_mask).sum() / num_valid

    return loss, clip_frac
