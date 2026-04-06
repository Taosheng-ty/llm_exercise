"""
Exercise 02: Sequence Packing - Solution
"""

import numpy as np


def pack_sequences(
    sequences: list[list[int]],
    max_seq_len: int,
    pad_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length sequences into fixed-length batch elements using FFD."""
    for i, seq in enumerate(sequences):
        if len(seq) > max_seq_len:
            raise ValueError(
                f"Sequence {i} has length {len(seq)} > max_seq_len {max_seq_len}"
            )

    if not sequences:
        return (
            np.zeros((0, max_seq_len), dtype=np.int64),
            np.zeros((0, max_seq_len), dtype=np.int64),
        )

    # Sort by length descending, keeping track of original indices
    indexed_seqs = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)

    # Bins: each bin is a list of token IDs accumulated so far
    bins: list[list[int]] = []
    bin_remaining: list[int] = []  # remaining capacity in each bin

    for _, seq in indexed_seqs:
        seq_len = len(seq)
        if seq_len == 0:
            continue

        # First-fit: find the first bin with enough room
        placed = False
        for b_idx in range(len(bins)):
            if bin_remaining[b_idx] >= seq_len:
                bins[b_idx].extend(seq)
                bin_remaining[b_idx] -= seq_len
                placed = True
                break

        if not placed:
            bins.append(list(seq))
            bin_remaining.append(max_seq_len - seq_len)

    if not bins:
        return (
            np.zeros((0, max_seq_len), dtype=np.int64),
            np.zeros((0, max_seq_len), dtype=np.int64),
        )

    num_bins = len(bins)
    packed_ids = np.full((num_bins, max_seq_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((num_bins, max_seq_len), dtype=np.int64)

    for b_idx, bin_tokens in enumerate(bins):
        length = len(bin_tokens)
        packed_ids[b_idx, :length] = bin_tokens
        attention_mask[b_idx, :length] = 1

    return packed_ids, attention_mask


def compute_packing_efficiency(
    sequences: list[list[int]],
    max_seq_len: int,
) -> float:
    """Compute the packing efficiency: ratio of real tokens to total slots."""
    packed_ids, _ = pack_sequences(sequences, max_seq_len)
    if packed_ids.shape[0] == 0:
        return 0.0
    total_real_tokens = sum(len(s) for s in sequences)
    total_slots = packed_ids.shape[0] * max_seq_len
    return total_real_tokens / total_slots
