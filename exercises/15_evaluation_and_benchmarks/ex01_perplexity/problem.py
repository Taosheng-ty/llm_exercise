"""
Exercise 1: Compute Perplexity of a Language Model

Perplexity (PPL) is a standard metric for evaluating language models. It measures
how "surprised" a model is by the data. Lower perplexity means better predictions.

Perplexity is the go-to metric for evaluating LLM quality — it quantifies how well
a model predicts held-out text by measuring the average negative log-likelihood per
token. It's used throughout LLM development to compare model sizes, training recipes,
and fine-tuning approaches, making it one of the first numbers reported in any
language modeling paper.

    PPL = exp( -1/N * sum(log_probs[i] * mask[i]) )

where N = sum(mask) is the number of valid (non-padding) tokens.

Your task: implement compute_perplexity() that handles masked sequences and
batch_perplexity() that computes per-sequence perplexity for a batch.

Difficulty: Medium
Framework: PyTorch
"""

import torch


def compute_perplexity(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from per-token log probabilities with masking.

    Args:
        log_probs: Tensor of shape (seq_len,) containing log probabilities
                   for each token (negative values, e.g., output of log_softmax
                   gathered at the correct token index).
        mask: Binary tensor of shape (seq_len,) where 1 indicates a valid token
              and 0 indicates padding that should be ignored.

    Returns:
        Scalar tensor with the perplexity value.
        If no valid tokens exist (mask is all zeros), return tensor(float('inf')).

    Formula:
        avg_neg_log_prob = -sum(log_probs * mask) / sum(mask)
        perplexity = exp(avg_neg_log_prob)

    Examples:
        >>> log_probs = torch.tensor([-1.0, -2.0, -3.0])
        >>> mask = torch.tensor([1.0, 1.0, 1.0])
        >>> compute_perplexity(log_probs, mask)  # exp(2.0) = 7.389...
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compute_perplexity")


def batch_perplexity(
    log_probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-sequence perplexity for a batch of sequences.

    Args:
        log_probs: Tensor of shape (batch_size, seq_len) containing log
                   probabilities for each token in each sequence.
        mask: Binary tensor of shape (batch_size, seq_len) indicating
              valid tokens.

    Returns:
        Tensor of shape (batch_size,) with perplexity for each sequence.
        Sequences with no valid tokens should have perplexity = inf.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement batch_perplexity")
