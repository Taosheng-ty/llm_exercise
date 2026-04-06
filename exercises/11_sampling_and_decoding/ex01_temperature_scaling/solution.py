"""Solution for Exercise 01: Temperature Scaling"""

import torch


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: (batch_size, vocab_size) raw logits from model
        temperature: float >= 0. 0 means greedy (argmax).

    Returns:
        Scaled logits of the same shape.
    """
    if temperature == 0:
        # Greedy: create a distribution that puts all mass on the argmax
        max_indices = logits.argmax(dim=-1, keepdim=True)
        scaled = torch.full_like(logits, float("-inf"))
        scaled.scatter_(-1, max_indices, 1e6)
        return scaled
    return logits / temperature
