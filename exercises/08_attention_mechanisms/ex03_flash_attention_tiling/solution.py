"""
Exercise 03: Flash Attention via Tiling - Solution
"""

import torch
import torch.nn.functional as F
import math


def flash_attention_tiling(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Tiled attention with online softmax trick.

    Process K/V in blocks, maintaining running max and sum for
    numerically stable softmax without materializing full NxN matrix.
    """
    B, H, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # Output accumulator
    O = torch.zeros_like(Q)
    # Running max per query position: (B, H, N, 1)
    m = torch.full((B, H, N, 1), float("-inf"), device=Q.device, dtype=Q.dtype)
    # Running sum of exponentials per query position: (B, H, N, 1)
    l = torch.zeros((B, H, N, 1), device=Q.device, dtype=Q.dtype)

    # Iterate over K/V blocks
    num_kv_blocks = (N + block_size - 1) // block_size
    for j in range(num_kv_blocks):
        j_start = j * block_size
        j_end = min(j_start + block_size, N)

        K_block = K[:, :, j_start:j_end, :]  # (B, H, block, D)
        V_block = V[:, :, j_start:j_end, :]

        # Compute scores for this block: (B, H, N, block)
        S_block = torch.matmul(Q, K_block.transpose(-2, -1)) * scale

        # Online softmax: find new max
        m_block = S_block.max(dim=-1, keepdim=True).values  # (B, H, N, 1)
        m_new = torch.maximum(m, m_block)

        # Rescale old accumulated values
        exp_old = torch.exp(m - m_new)
        # Compute new block exponentials
        exp_block = torch.exp(S_block - m_new)

        # Update running sum
        l_new = l * exp_old + exp_block.sum(dim=-1, keepdim=True)

        # Update output: rescale old output and add new contribution
        O = O * exp_old + torch.matmul(exp_block, V_block)

        m = m_new
        l = l_new

    # Normalize
    O = O / l
    return O
