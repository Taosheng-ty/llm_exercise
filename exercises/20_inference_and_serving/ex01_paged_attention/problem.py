"""
Exercise 01: Paged Attention KV Cache (Hard)

In modern LLM serving systems (e.g., vLLM), the key-value cache is managed using
paged memory, analogous to virtual memory in operating systems. Instead of allocating
one contiguous buffer per sequence, memory is divided into fixed-size blocks that can
be allocated on demand and shared across sequences.

This exercise implements a simplified paged KV cache:

Key concepts:
- The KV cache is divided into `num_blocks` blocks, each holding `block_size` tokens.
- Each sequence maintains a "block table" -- a list of block indices mapping logical
  positions to physical blocks.
- Tokens are appended one at a time; when a block fills up, a new one is allocated.
- Reading back the KV cache reconstructs the contiguous key/value tensors from the
  scattered blocks.

Your tasks:
-----------
1. Implement `PagedKVCache`:

   - __init__(self, num_blocks, block_size, num_heads, head_dim):
       Allocate the physical KV storage as two tensors:
         self.key_cache: shape (num_blocks, block_size, num_heads, head_dim)
         self.value_cache: shape (num_blocks, block_size, num_heads, head_dim)
       Track which blocks are free and how many tokens are filled in each block.

   - allocate_blocks(self, num_needed) -> list[int]:
       Allocate `num_needed` free blocks. Return their indices.
       Raise RuntimeError if not enough free blocks.

   - free_blocks(self, block_indices):
       Return blocks to the free pool.

   - append_token(self, block_table, key, value) -> list[int]:
       Append a single token's key and value to the sequence described by block_table.
       key, value: shape (num_heads, head_dim).
       If the last block in the table is full, allocate a new block.
       Return the (possibly extended) block_table.

   - read_kv(self, block_table, seq_len) -> tuple[Tensor, Tensor]:
       Read back keys and values for a sequence of length `seq_len`.
       Returns (keys, values) each of shape (seq_len, num_heads, head_dim).

   - num_free_blocks(self) -> int:
       Return count of currently free blocks.
"""

import torch


class PagedKVCache:
    """Simplified paged KV cache with fixed-size blocks."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
    ):
        raise NotImplementedError("Implement PagedKVCache.__init__")

    def allocate_blocks(self, num_needed: int) -> list:
        """Allocate num_needed free blocks. Return list of block indices."""
        raise NotImplementedError("Implement allocate_blocks")

    def free_blocks(self, block_indices: list) -> None:
        """Return blocks to the free pool."""
        raise NotImplementedError("Implement free_blocks")

    def append_token(self, block_table: list, key: torch.Tensor, value: torch.Tensor) -> list:
        """
        Append one token's KV to the sequence. Returns updated block_table.

        Args:
            block_table: list of block indices for this sequence
            key: shape (num_heads, head_dim)
            value: shape (num_heads, head_dim)
        """
        raise NotImplementedError("Implement append_token")

    def read_kv(self, block_table: list, seq_len: int):
        """
        Read keys and values for a sequence.

        Returns:
            (keys, values) each of shape (seq_len, num_heads, head_dim)
        """
        raise NotImplementedError("Implement read_kv")

    def num_free_blocks(self) -> int:
        """Return the number of currently free blocks."""
        raise NotImplementedError("Implement num_free_blocks")
