"""
Solution for Exercise 04: SFT Training Step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked cross-entropy loss for SFT."""
    # Shift: logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous()

    B, T, V = shift_logits.shape
    # Flatten for cross_entropy
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, T)

    # Apply mask and average
    masked_loss = per_token_loss * shift_mask
    num_tokens = shift_mask.sum().clamp(min=1.0)
    return masked_loss.sum() / num_tokens


def sft_training_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Execute one SFT training step."""
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = compute_sft_loss(logits, input_ids, loss_mask)
    loss.backward()
    optimizer.step()
    return loss.item()
