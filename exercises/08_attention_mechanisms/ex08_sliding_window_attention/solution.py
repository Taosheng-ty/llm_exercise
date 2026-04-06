"""
Exercise 08: Sliding Window Attention - Solution
"""

import torch
import torch.nn.functional as F
import math


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a sliding window mask.

    Token i can attend to tokens [max(0, i-window_size+1), i].
    Returns bool mask where True = masked (outside window).
    """
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

    # Token i attends to j if: j <= i (causal) AND j >= i - window_size + 1
    causal = cols <= rows
    in_window = cols >= (rows - window_size + 1)

    # Mask is True where we should NOT attend
    mask = ~(causal & in_window)
    return mask


def sliding_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sliding window attention: each token attends only to the
    previous window_size tokens (including itself).
    """
    seq_len = Q.size(2)
    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    mask = create_sliding_window_mask(seq_len, window_size, device=Q.device)
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, attn
