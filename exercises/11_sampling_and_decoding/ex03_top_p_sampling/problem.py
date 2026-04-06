"""Exercise 03: Top-P (Nucleus) Sampling (Medium)

Top-p sampling (also called nucleus sampling) dynamically selects the smallest
set of tokens whose cumulative probability exceeds p. This adapts the number
of candidate tokens based on the shape of the distribution.

Algorithm:
1. Sort logits in descending order
2. Compute softmax probabilities of the sorted logits
3. Compute cumulative sum of these probabilities
4. Create a mask for tokens where cumulative prob exceeds p
5. Set masked logits to -inf (but always keep at least the top-1 token)
6. Scatter the filtered values back to original positions

Implement `top_p_filter(logits, p)`:

Args:
    logits: torch.Tensor of shape (batch_size, vocab_size)
    p: float in (0, 1] — cumulative probability threshold

Returns:
    torch.Tensor of same shape — filtered logits
"""

import torch


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    # TODO: Implement nucleus (top-p) sampling filter
    raise NotImplementedError("Implement top_p_filter")
