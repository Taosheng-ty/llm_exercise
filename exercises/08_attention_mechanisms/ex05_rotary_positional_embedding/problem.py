"""
Exercise 05: Rotary Positional Embedding (RoPE)
Difficulty: Hard

Implement RoPE which encodes position information by rotating pairs of
dimensions in Q and K vectors.

Implement two functions:

1. compute_rope_frequencies(head_dim, max_seq_len, base=10000.0)
   - Compute the sinusoidal frequency matrix
   - For dimension pair i, freq = 1 / (base^(2i/head_dim))
   - Returns cos and sin tensors of shape (max_seq_len, head_dim)
   - Note: each dimension pair shares the same frequency, so the returned
     tensors have repeated values: cos[:, 2i] == cos[:, 2i+1] and likewise
     for sin. (i.e., expand from half_dim to head_dim by repeating each
     frequency value for both dimensions in the pair.)

2. apply_rope(x, cos, sin)
   - x: (batch, heads, seq_len, head_dim)
   - Rotate pairs of dimensions: for each pair (x_2i, x_{2i+1}):
       out_2i   = x_2i * cos - x_{2i+1} * sin
       out_2i+1 = x_2i * sin + x_{2i+1} * cos
   - Returns rotated tensor of same shape
"""

import torch


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Compute cos and sin frequency tensors
    raise NotImplementedError


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # TODO: Apply rotary embeddings to x
    raise NotImplementedError
