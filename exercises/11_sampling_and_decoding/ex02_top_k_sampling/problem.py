"""Exercise 02: Top-K Sampling (Easy)

Top-k sampling restricts the candidate tokens to the k most probable ones.
All other tokens have their logits set to -inf so they receive zero probability
after softmax.

Implement `top_k_filter(logits, k)` that:
1. For each row in the batch, find the k-th largest logit value
2. Set all logits below that threshold to -inf
3. Return the filtered logits (do NOT apply softmax)

Special cases:
- k=1 is equivalent to greedy decoding (only the top token survives)
- k >= vocab_size means no filtering

Args:
    logits: torch.Tensor of shape (batch_size, vocab_size)
    k: int — number of top tokens to keep

Returns:
    torch.Tensor of same shape — filtered logits
"""

import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    # TODO: Implement top-k filtering
    # Hint: torch.topk can help find the k-th largest value
    raise NotImplementedError("Implement top_k_filter")
