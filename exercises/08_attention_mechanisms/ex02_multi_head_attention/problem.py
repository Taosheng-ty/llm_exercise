"""
Exercise 02: Multi-Head Attention
Difficulty: Medium

Implement multi-head attention as a PyTorch module. Multi-head attention
allows LLMs to attend to different representation subspaces simultaneously
-- e.g., one head may track syntax while another tracks semantics. This
parallelism is what gives transformer LLMs their expressive power for
language understanding and generation.

The module should:
  1. Project input x into Q, K, V using linear layers
  2. Split into multiple heads: head_dim = d_model // num_heads
  3. Compute scaled dot-product attention per head
  4. Concatenate heads and apply output projection

Args (forward):
    x: Input tensor of shape (batch, seq_len, d_model)
    causal: If True, apply causal mask

Returns:
    output: Tensor of shape (batch, seq_len, d_model)
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: Initialize projection layers and parameters
        raise NotImplementedError

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        # TODO: Implement multi-head attention forward pass
        raise NotImplementedError
