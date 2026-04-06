"""
Exercise 08: Sliding Window Attention
Difficulty: Medium

Implement sliding window attention where each token only attends
to the previous `window_size` tokens (including itself).

Implement two functions:

1. create_sliding_window_mask(seq_len, window_size)
   - Returns boolean mask of shape (seq_len, seq_len)
   - True = masked (token outside the window)
   - Token i attends to tokens max(0, i-window_size+1) through i

2. sliding_window_attention(Q, K, V, window_size)
   - Q, K, V: (batch, heads, seq_len, head_dim)
   - Apply sliding window mask and compute attention
   - Returns output (batch, heads, seq_len, head_dim)
     and attention weights (batch, heads, seq_len, seq_len)
"""

import torch


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    # TODO: Create sliding window mask
    raise NotImplementedError


def sliding_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Implement sliding window attention
    raise NotImplementedError
