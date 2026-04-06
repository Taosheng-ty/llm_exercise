"""
Solution for Exercise 02: Extract Per-Token Log Probabilities from Logits
"""
import numpy as np


def log_probs_from_logits(
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> np.ndarray:
    """Extract per-token log probabilities for chosen tokens.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size).
        token_ids: Chosen token IDs of shape (batch, seq_len), integer-valued.

    Returns:
        Array of shape (batch, seq_len) containing the log probability of
        each token in token_ids under the distribution defined by logits.
    """
    # Step 1: Numerically stable log-softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_softmax = shifted - log_sum_exp  # (batch, seq_len, vocab_size)

    # Step 2: Gather log probs for the chosen tokens
    batch_size, seq_len, _ = logits.shape
    batch_idx = np.arange(batch_size)[:, None]
    seq_idx = np.arange(seq_len)[None, :]
    return log_softmax[batch_idx, seq_idx, token_ids]  # (batch, seq_len)
