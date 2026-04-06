"""
Exercise 02: Sequence Packing (Medium)

In LLM training, sequences have variable lengths but GPU batches require fixed sizes.
Naive padding wastes compute on padding tokens. Sequence packing solves this by fitting
multiple shorter sequences into a single batch slot, reducing wasted computation.

This exercise implements a greedy first-fit-decreasing (FFD) bin-packing algorithm:
1. Sort sequences by length (longest first)
2. For each sequence, try to fit it into the first bin that has enough room
3. If no bin fits, create a new bin

Each "bin" is a packed batch element of max_seq_len tokens.

Reference: Common LLM training optimization used in systems like slime.
"""

import numpy as np


def pack_sequences(
    sequences: list[list[int]],
    max_seq_len: int,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length sequences into fixed-length batch elements using FFD.

    Uses a greedy first-fit-decreasing bin-packing algorithm:
    1. Sort sequences by length in decreasing order.
    2. For each sequence, place it in the first bin with enough remaining space.
    3. If no bin has room, create a new bin.
    4. Pad each bin to max_seq_len with pad_token_id.

    Args:
        sequences: List of token ID lists, each of variable length.
            All sequences must have length <= max_seq_len.
        max_seq_len: Maximum sequence length for each packed batch element.
        pad_token_id: Token ID used for padding. Defaults to 0.

    Returns:
        A tuple of (packed_ids, attention_mask):
        - packed_ids: np.ndarray of shape (num_bins, max_seq_len) with packed token IDs.
        - attention_mask: np.ndarray of shape (num_bins, max_seq_len) with 1 for real
          tokens and 0 for padding.

    Raises:
        ValueError: If any sequence is longer than max_seq_len.

    Example:
        >>> seqs = [[1, 2, 3], [4, 5], [6]]
        >>> ids, mask = pack_sequences(seqs, max_seq_len=5)
        >>> # Could pack [1,2,3] + [4,5] into one bin and [6] into another
    """
    # TODO: Implement this function
    raise NotImplementedError


def compute_packing_efficiency(
    sequences: list[list[int]],
    max_seq_len: int,
) -> float:
    """Compute the packing efficiency: ratio of real tokens to total slots.

    Efficiency = total_real_tokens / (num_bins * max_seq_len)

    Args:
        sequences: List of token ID lists.
        max_seq_len: Maximum sequence length per bin.

    Returns:
        A float between 0.0 and 1.0 representing packing efficiency.
    """
    # TODO: Implement this function
    raise NotImplementedError
