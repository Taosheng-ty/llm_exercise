"""
Exercise 03: Flash Attention via Tiling
Difficulty: Hard

Implement a simplified flash attention using tiling to reduce memory usage.
Flash attention is a critical optimization for LLM training -- standard
attention has O(n^2) memory in sequence length, which limits context windows
and batch sizes. By computing attention in tiles without materializing the
full attention matrix, flash attention enables training with longer sequences
and larger batches on the same GPU memory.

Instead of materializing the full N x N attention matrix, process Q and K/V
in blocks. Use the online softmax trick to maintain numerical stability:
  - Track running max (m) and sum (l) across blocks
  - Rescale accumulated output when the max changes:
      When processing a new block with local max m_new:
        m_combined = max(m_old, m_new)
        l_old_rescaled = l_old * exp(m_old - m_combined)
        l_new = sum(exp(scores - m_combined))   [for the new block]
        O = (O * l_old_rescaled + exp(scores - m_combined) @ V_block) / (l_old_rescaled + l_new)
        l = l_old_rescaled + l_new
        m = m_combined

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
