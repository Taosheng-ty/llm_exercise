"""
Exercise 05: Rotary Positional Embedding (RoPE) - Solution
"""

import torch


def compute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos and sin frequency tables for RoPE.

    Uses the interleaved convention: dimension pairs are (0,1), (2,3), etc.

    Returns:
        cos: (max_seq_len, head_dim) - cosines for each dimension
        sin: (max_seq_len, head_dim) - sines for each dimension
    """
    assert head_dim % 2 == 0, "head_dim must be even"
    half_dim = head_dim // 2
    # Frequencies for each pair: 1/base^(2i/d) for i in [0, head_dim/2)
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))

    # Position indices
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # Outer product: (max_seq_len, half_dim)
    angles = torch.outer(positions, freqs)

    # Interleave: each freq appears twice (for the pair of dims it rotates)
    # angles_full[..., 0] = angles_full[..., 1] = angles[..., 0], etc.
    angles_full = angles.repeat_interleave(2, dim=-1)  # (max_seq_len, head_dim)

    cos = torch.cos(angles_full)
    sin = torch.sin(angles_full)

    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding to x.

    x: (batch, heads, seq_len, head_dim)
    cos, sin: (max_seq_len, head_dim) precomputed frequencies

    For interleaved pairs (x_2i, x_{2i+1}):
        out_2i   = x_2i * cos - x_{2i+1} * sin
        out_2i+1 = x_{2i+1} * cos + x_2i * sin
    """
    seq_len = x.size(2)

    # Slice to actual sequence length and reshape for broadcasting
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Build the "rotated" companion: swap each pair and negate the first
    # For pair (x0, x1) -> (-x1, x0)
    x_paired = x.reshape(*x.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    x_rotated = torch.stack([-x_paired[..., 1], x_paired[..., 0]], dim=-1)
    x_rotated = x_rotated.reshape(x.shape)

    return x * cos + x_rotated * sin
