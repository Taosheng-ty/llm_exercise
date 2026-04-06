"""
Exercise 01: Scaled Dot-Product Attention
Difficulty: Medium

Implement scaled dot-product attention from scratch.

Given Q, K, V tensors of shape (batch, heads, seq_len, head_dim):
  1. Compute attention scores: score = Q @ K^T / sqrt(d_k)
  2. Optionally apply a causal mask (set future positions to -inf)
  3. Apply softmax over the last dimension
  4. Compute output = attention_weights @ V

Args:
    Q: Query tensor of shape (batch, heads, seq_len, head_dim)
    K: Key tensor of shape (batch, heads, seq_len, head_dim)
    V: Value tensor of shape (batch, heads, seq_len, head_dim)
    causal: If True, apply a causal (lower-triangular) mask

Returns:
    output: Tensor of shape (batch, heads, seq_len, head_dim)
    attention_weights: Tensor of shape (batch, heads, seq_len, seq_len)
"""

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Implement scaled dot-product attention
    raise NotImplementedError
