"""
Exercise 04: Causal Mask
Difficulty: Easy

Generate and apply causal (autoregressive) attention masks.

Implement two functions:

1. create_causal_mask(seq_len, batch_broadcastable=False)
   - Returns a boolean mask where True = masked (future positions)
   - If batch_broadcastable=False: shape (seq_len, seq_len)
   - If batch_broadcastable=True: shape (1, 1, seq_len, seq_len)

2. apply_causal_mask(scores, mask)
   - scores: attention scores tensor
   - mask: boolean mask (True = positions to mask)
   - Set masked positions to -inf
   - Return modified scores
"""

import torch


def create_causal_mask(
    seq_len: int,
    batch_broadcastable: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    # TODO: Create a causal mask
    raise NotImplementedError


def apply_causal_mask(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # TODO: Apply the causal mask to attention scores
    raise NotImplementedError
