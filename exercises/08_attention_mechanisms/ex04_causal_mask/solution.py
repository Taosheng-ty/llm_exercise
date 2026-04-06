"""
Exercise 04: Causal Mask - Solution
"""

import torch


def create_causal_mask(
    seq_len: int,
    batch_broadcastable: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a causal mask where True indicates positions to mask (future tokens).

    Returns upper-triangular boolean mask (diagonal=1).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    if batch_broadcastable:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    return mask


def apply_causal_mask(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply causal mask to attention scores by setting masked positions to -inf.
    """
    return scores.masked_fill(mask, float("-inf"))
