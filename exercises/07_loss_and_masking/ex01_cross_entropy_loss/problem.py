"""
Exercise 01: Cross-Entropy Loss for Language Modeling

In language model training (SFT and RL), we need to compute the cross-entropy
loss between the model's predicted logits and the target token IDs. A loss mask
is applied so that only response tokens contribute to the loss (prompt tokens
are masked out).

Your task:
  1. Implement numerically stable log-softmax (subtract max before exp).
  2. Compute per-token cross-entropy = -log_softmax[target].
  3. Apply the loss mask and return the mean loss over unmasked tokens.

Reference: This is the standard language modeling loss used throughout the
slime training pipeline (SFT loss, policy gradient baselines, etc.).

Difficulty: Medium
"""
import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable log-softmax along the last axis.

    Args:
        logits: Array of shape (..., vocab_size).

    Returns:
        log_softmax values with the same shape as logits.
    """
    # TODO: Implement numerically stable log-softmax
    #   Hint: subtract the max along the last axis for numerical stability.
    raise NotImplementedError


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray,
) -> float:
    """Compute masked cross-entropy loss for language modeling.

    Args:
        logits: Predicted logits of shape (batch, seq_len, vocab_size).
        targets: Target token IDs of shape (batch, seq_len), integer-valued.
        loss_mask: Binary mask of shape (batch, seq_len). 1 = compute loss,
                   0 = ignore this position.

    Returns:
        Scalar mean cross-entropy loss over all unmasked positions.
        If no positions are unmasked, return 0.0.
    """
    # TODO: Implement cross-entropy loss
    #   1. Compute log_softmax of logits along the vocab dimension.
    #   2. Gather the log probability of the target token at each position.
    #   3. Compute per-token cross-entropy = -log_prob_of_target.
    #   4. Apply loss_mask and return mean over unmasked tokens.
    raise NotImplementedError
