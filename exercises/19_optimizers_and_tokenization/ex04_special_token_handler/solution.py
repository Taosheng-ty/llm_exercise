"""Solution for Exercise 04: Special Token Handler"""

import numpy as np


def encode_with_specials(
    token_ids: list[int],
    bos_id: int,
    eos_id: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[int]:
    """Add BOS and/or EOS special tokens to a sequence."""
    result = list(token_ids)
    if add_bos:
        result = [bos_id] + result
    if add_eos:
        result = result + [eos_id]
    return result


def pad_batch(
    sequences: list[list[int]],
    pad_id: int,
    padding_side: str = "right",
    max_len: int | None = None,
) -> np.ndarray:
    """Pad a batch of variable-length sequences to uniform length."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    batch = np.full((len(sequences), max_len), pad_id, dtype=np.int64)

    for i, seq in enumerate(sequences):
        # Truncate if needed
        truncated = seq[:max_len]
        seq_len = len(truncated)
        if padding_side == "right":
            batch[i, :seq_len] = truncated
        else:  # left padding
            batch[i, max_len - seq_len :] = truncated

    return batch


def create_attention_mask(padded_ids: np.ndarray, pad_id: int) -> np.ndarray:
    """Create an attention mask from padded token IDs."""
    return (padded_ids != pad_id).astype(np.int64)
