"""
Exercise 03: ALiBi Attention (Medium)

ALiBi (Attention with Linear Biases) is a positional encoding method from
"Train Short, Test Long" (Press et al., 2022). Instead of adding positional
embeddings to token representations, ALiBi adds a static, non-learned bias
to the attention scores that penalizes distant tokens linearly.

Key ideas:
- Each attention head gets a "slope" m_i from a geometric sequence.
- The bias for head i at query position q and key position k is: m_i * (k - q)
  (this is <= 0 for causal attention since k <= q).
- This allows extrapolation to longer sequences than seen during training.

Your tasks:
-----------
1. Implement `compute_alibi_slopes(num_heads) -> Tensor`:
   - Returns a 1D tensor of slopes of length num_heads.
   - Slopes follow a geometric sequence: start = 2^(-8/num_heads), ratio = start.
   - So: slopes[i] = 2^(-8 * (i+1) / num_heads) for i in 0..num_heads-1.

2. Implement `compute_alibi_bias(seq_len, num_heads) -> Tensor`:
   - Returns tensor of shape (num_heads, seq_len, seq_len).
   - bias[h, q, k] = slopes[h] * (k - q).
   - For causal masking, positions where k > q can be set to -inf (handled
     separately) or just computed -- the causal mask handles it.

3. Implement `alibi_attention(Q, K, V, alibi_bias, causal_mask=True) -> Tensor`:
   - Q, K, V: shape (batch, num_heads, seq_len, head_dim)
   - alibi_bias: shape (num_heads, seq_len, seq_len) or (1, num_heads, seq_len, seq_len)
   - Compute scaled dot-product attention with ALiBi bias added to scores.
   - If causal_mask is True, mask out future positions with -inf.
   - Returns output of shape (batch, num_heads, seq_len, head_dim).
"""

import torch


def compute_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes as a geometric sequence.

    Returns:
        Tensor of shape (num_heads,)
    """
    raise NotImplementedError("Implement compute_alibi_slopes")


def compute_alibi_bias(seq_len: int, num_heads: int) -> torch.Tensor:
    """
    Compute the ALiBi distance bias matrix.

    Returns:
        Tensor of shape (num_heads, seq_len, seq_len)
    """
    raise NotImplementedError("Implement compute_alibi_bias")


def alibi_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    alibi_bias: torch.Tensor,
    causal_mask: bool = True,
) -> torch.Tensor:
    """
    Scaled dot-product attention with ALiBi positional bias.

    Args:
        Q: (batch, num_heads, seq_len, head_dim)
        K: (batch, num_heads, seq_len, head_dim)
        V: (batch, num_heads, seq_len, head_dim)
        alibi_bias: (num_heads, seq_len, seq_len) or broadcastable
        causal_mask: whether to apply causal masking

    Returns:
        Output of shape (batch, num_heads, seq_len, head_dim)
    """
    raise NotImplementedError("Implement alibi_attention")
