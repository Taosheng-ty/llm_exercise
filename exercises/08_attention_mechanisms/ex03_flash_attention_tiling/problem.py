"""
Exercise 03: Flash Attention via Tiling
Difficulty: Hard

Implement a simplified flash attention using tiling to reduce memory usage.

Instead of materializing the full N x N attention matrix, process Q and K/V
in blocks. Use the online softmax trick to maintain numerical stability:
  - Track running max (m) and sum (l) across blocks
  - Rescale accumulated output when the max changes

Args:
    Q: (batch, heads, seq_len, head_dim)
    K: (batch, heads, seq_len, head_dim)
    V: (batch, heads, seq_len, head_dim)
    block_size: Number of tokens per block

Returns:
    output: (batch, heads, seq_len, head_dim) - same as standard attention
"""

import torch


def flash_attention_tiling(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    # TODO: Implement tiled flash attention with online softmax
    raise NotImplementedError
