"""
Exercise 03: Clipped Value Function Loss
==========================================

Implement the PPO-style clipped value function loss used for critic training.

Given:
- values:     (batch, seq_len) current value predictions (requires grad)
- old_values: (batch, seq_len) value predictions from the rollout policy
- returns:    (batch, seq_len) target returns (from GAE)
- loss_mask:  (batch, seq_len) binary mask
- value_clip: clipping threshold (e.g., 0.2)

Compute:
1. values_clipped = old_values + clamp(values - old_values, -value_clip, +value_clip)
2. surr1 = (values_clipped - returns)^2
3. surr2 = (values - returns)^2
4. per-token loss = max(surr1, surr2)
5. Return masked mean loss and clip fraction.

Reference: slime/backends/megatron_utils/loss.py :: value_loss_function()
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
    # TODO: Implement clipped value loss
    # Hint 1: Clip the value change relative to old_values
    # Hint 2: Compute squared error for both clipped and unclipped
    # Hint 3: Take the maximum (pessimistic)
    raise NotImplementedError("Implement compute_value_loss")
