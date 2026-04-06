"""
Solution for Exercise 01: Cross-Entropy Loss for Language Modeling
"""
import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable log-softmax along the last axis.

    Args:
        logits: Array of shape (..., vocab_size).

    Returns:
        log_softmax values with the same shape as logits.
    """
    # Subtract max for numerical stability
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


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
    batch_size, seq_len, vocab_size = logits.shape

    # Step 1: Compute log-softmax
    log_probs = log_softmax(logits)

    # Step 2: Gather log probabilities of target tokens
    # Use advanced indexing: for each (b, t), pick log_probs[b, t, targets[b, t]]
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_idx, seq_idx, targets]  # (batch, seq_len)

    # Step 3: Per-token cross-entropy
    per_token_loss = -target_log_probs

    # Step 4: Apply mask and compute mean
    masked_loss = per_token_loss * loss_mask
    num_unmasked = loss_mask.sum()

    if num_unmasked == 0:
        return 0.0

    return float(masked_loss.sum() / num_unmasked)
