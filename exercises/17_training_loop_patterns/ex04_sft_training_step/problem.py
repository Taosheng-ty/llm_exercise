"""
Exercise 04: SFT Training Step

Difficulty: Medium
Framework: PyTorch

Background:
    Supervised Fine-Tuning (SFT) trains a language model to imitate
    demonstrations. The key pattern:

    1. Forward pass: model(input_ids) -> logits
    2. Compute cross-entropy loss on next-token prediction
    3. Apply loss mask: only compute loss on assistant/response tokens,
       not on prompt tokens
    4. Backward pass + optimizer step

    Reference: slime/rollout/sft_rollout.py generates token_ids and loss_mask
    pairs where loss_mask indicates which tokens contribute to the loss.

Implement:
    compute_sft_loss(logits, labels, loss_mask) -> scalar loss
    sft_training_step(model, input_ids, loss_mask, optimizer) -> loss value

Args for compute_sft_loss:
    logits: (batch, seq_len, vocab_size) - model output logits
    labels: (batch, seq_len) - target token ids (shifted input_ids)
    loss_mask: (batch, seq_len) - binary mask, 1 for tokens that count

Args for sft_training_step:
    model: nn.Module that takes input_ids and returns logits
    input_ids: (batch, seq_len) - input token ids
    loss_mask: (batch, seq_len) - binary mask for which positions to train on
    optimizer: torch.optim.Optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked cross-entropy loss for SFT.

    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target token ids
        loss_mask: (B, T) binary mask (1 = compute loss, 0 = ignore)

    Returns:
        Scalar loss averaged over masked positions.
    """
    # TODO: Implement masked cross-entropy
    # Hint 1: Use F.cross_entropy with reduction='none' to get per-token loss
    # Hint 2: logits for prediction are logits[:, :-1, :] predicting labels[:, 1:]
    # Hint 3: Multiply per-token loss by loss_mask[:, 1:] (shift mask too)
    # Hint 4: Average over masked positions: sum(masked_loss) / sum(mask)
    raise NotImplementedError("Implement compute_sft_loss")


def sft_training_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Execute one SFT training step.

    Args:
        model: language model that returns logits given input_ids
        input_ids: (B, T) input token ids
        loss_mask: (B, T) binary mask
        optimizer: optimizer instance

    Returns:
        Loss value as a Python float.
    """
    # TODO: Implement forward + backward + step
    # Hint 1: logits = model(input_ids), labels = input_ids
    # Hint 2: loss = compute_sft_loss(logits, input_ids, loss_mask)
    # Hint 3: optimizer.zero_grad(), loss.backward(), optimizer.step()
    raise NotImplementedError("Implement sft_training_step")
