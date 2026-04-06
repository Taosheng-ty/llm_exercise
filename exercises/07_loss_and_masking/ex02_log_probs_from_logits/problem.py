"""
Exercise 02: Extract Per-Token Log Probabilities from Logits

In PPO and other policy gradient methods, we need to compute log pi(a|s) --
the log probability of each generated token under the current policy. This is
done by applying log-softmax to the logits and then "gathering" the log prob
corresponding to each chosen token.

Your task:
  1. Compute log-softmax of the logits (numerically stable).
  2. Gather the log probability for each token in token_ids.
  3. Return the per-token log probabilities.

Reference: slime's calculate_log_probs_and_entropy() and compute_log_probs()
in ppo_utils.py.

Difficulty: Easy
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
    # TODO: Implement this function
    #   1. Compute numerically stable log-softmax along the vocab dimension.
    #   2. Use advanced indexing to gather log probs for token_ids.
    raise NotImplementedError
