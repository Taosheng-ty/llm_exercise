"""
Solution for Exercise 05: Multi-Token Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenPredictionHead(nn.Module):
    """N-ahead prediction heads from a single hidden state."""

    def __init__(self, d_model: int, vocab_size: int, num_futures: int):
        super().__init__()
        self.num_futures = num_futures
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(num_futures)]
        )

    def forward(self, hidden_states: torch.Tensor) -> list:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            List of num_futures tensors, each (batch, seq_len, vocab_size)
        """
        return [head(hidden_states) for head in self.heads]


def compute_multi_token_loss(
    predictions: list,
    targets: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Compute cross-entropy loss for each prediction head.

    For head i (0-indexed), at position t, the target is targets[:, t + i + 1].
    We only compute loss where mask is True and the shifted target position is valid.

    Args:
        predictions: list of N tensors, each (batch, seq_len, vocab_size)
        targets: (batch, seq_len) token IDs
        mask: (batch, seq_len) boolean mask

    Returns:
        (total_loss, per_head_losses) where total_loss is scalar mean,
        per_head_losses is list of N scalar losses.
    """
    num_futures = len(predictions)
    batch_size, seq_len = targets.shape
    per_head_losses = []

    for i in range(num_futures):
        shift = i + 1
        # The prediction at position t targets token at position t + shift
        # Valid positions: t where t + shift < seq_len
        max_t = seq_len - shift
        if max_t <= 0:
            per_head_losses.append(torch.tensor(0.0, device=targets.device))
            continue

        # Truncate predictions and targets to valid range
        pred_i = predictions[i][:, :max_t, :]  # (B, max_t, V)
        target_i = targets[:, shift : shift + max_t]  # (B, max_t)
        mask_i = mask[:, :max_t]  # (B, max_t)

        # Flatten for cross_entropy
        pred_flat = pred_i.reshape(-1, pred_i.size(-1))  # (B*max_t, V)
        target_flat = target_i.reshape(-1)  # (B*max_t,)
        mask_flat = mask_i.reshape(-1)  # (B*max_t,)

        # Compute per-element loss
        loss_flat = F.cross_entropy(pred_flat, target_flat, reduction="none")  # (B*max_t,)

        # Apply mask
        if mask_flat.any():
            head_loss = (loss_flat * mask_flat.float()).sum() / mask_flat.float().sum()
        else:
            head_loss = torch.tensor(0.0, device=targets.device)

        per_head_losses.append(head_loss)

    total_loss = sum(per_head_losses) / max(len([l for l in per_head_losses if l.item() > 0]), 1)

    return total_loss, per_head_losses
