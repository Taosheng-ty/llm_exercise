"""Solution for Exercise 02: Top-K Sampling"""

import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Filter logits to keep only the top-k highest values.

    Args:
        logits: (batch_size, vocab_size) raw logits
        k: number of top tokens to keep

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    vocab_size = logits.size(-1)
    if k >= vocab_size:
        return logits

    # Find the k-th largest value per row
    # topk returns (values, indices); we need the minimum of top-k values
    top_k_values, _ = logits.topk(k, dim=-1)  # (batch_size, k)
    threshold = top_k_values[:, -1:]  # (batch_size, 1) — the k-th largest

    # Mask out everything below the threshold
    filtered = logits.clone()
    filtered[filtered < threshold] = float("-inf")
    return filtered
