"""
Solution for Exercise 01: Paged Attention KV Cache
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
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical storage: (num_blocks, block_size, num_heads, head_dim)
        self.key_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)
        self.value_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim)

        # Track free blocks (set of indices) and per-block fill counts
        self.free_set = set(range(num_blocks))
        self.block_fill = [0] * num_blocks  # how many tokens written in each block

    def allocate_blocks(self, num_needed: int) -> list:
        """Allocate num_needed free blocks. Return list of block indices."""
        if num_needed > len(self.free_set):
            raise RuntimeError(
                f"Cannot allocate {num_needed} blocks, only {len(self.free_set)} free"
            )
        allocated = []
        for _ in range(num_needed):
            block_idx = self.free_set.pop()
            self.block_fill[block_idx] = 0
            allocated.append(block_idx)
        return allocated

    def free_blocks(self, block_indices: list) -> None:
        """Return blocks to the free pool."""
        for idx in block_indices:
            self.block_fill[idx] = 0
            self.key_cache[idx].zero_()
            self.value_cache[idx].zero_()
            self.free_set.add(idx)

    def append_token(self, block_table: list, key: torch.Tensor, value: torch.Tensor) -> list:
        """
        Append one token's KV to the sequence. Returns updated block_table.

        Args:
            block_table: list of block indices for this sequence
            key: shape (num_heads, head_dim)
            value: shape (num_heads, head_dim)
        """
        # If no blocks yet or last block is full, allocate a new one
        if len(block_table) == 0 or self.block_fill[block_table[-1]] >= self.block_size:
            new_blocks = self.allocate_blocks(1)
            block_table = block_table + new_blocks

        # Write into the last block at the current fill position
        last_block = block_table[-1]
        pos = self.block_fill[last_block]
        self.key_cache[last_block, pos] = key
        self.value_cache[last_block, pos] = value
        self.block_fill[last_block] = pos + 1

        return block_table

    def read_kv(self, block_table: list, seq_len: int):
        """
        Read keys and values for a sequence.

        Returns:
            (keys, values) each of shape (seq_len, num_heads, head_dim)
        """
        keys = torch.zeros(seq_len, self.num_heads, self.head_dim)
        values = torch.zeros(seq_len, self.num_heads, self.head_dim)

        token_idx = 0
        for block_idx in block_table:
            tokens_in_block = min(self.block_size, seq_len - token_idx)
            if tokens_in_block <= 0:
                break
            keys[token_idx : token_idx + tokens_in_block] = self.key_cache[
                block_idx, :tokens_in_block
            ]
            values[token_idx : token_idx + tokens_in_block] = self.value_cache[
                block_idx, :tokens_in_block
            ]
            token_idx += tokens_in_block

        return keys, values

    def num_free_blocks(self) -> int:
        """Return the number of currently free blocks."""
        return len(self.free_set)
