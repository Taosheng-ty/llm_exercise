"""
Solution for Exercise 1: Compute Perplexity of a Language Model
"""

import torch


def compute_perplexity(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from per-token log probabilities with masking."""
    mask = mask.float()
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(float("inf"))

    avg_neg_log_prob = -(log_probs * mask).sum() / num_valid
    return torch.exp(avg_neg_log_prob)


def batch_perplexity(
    log_probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-sequence perplexity for a batch of sequences."""
    mask = mask.float()
    num_valid = mask.sum(dim=1)  # (batch_size,)
    sum_log_probs = (log_probs * mask).sum(dim=1)  # (batch_size,)

    avg_neg_log_prob = -sum_log_probs / num_valid.clamp(min=1e-12)
    ppl = torch.exp(avg_neg_log_prob)

    # Set inf for sequences with no valid tokens
    ppl[num_valid == 0] = float("inf")
    return ppl
