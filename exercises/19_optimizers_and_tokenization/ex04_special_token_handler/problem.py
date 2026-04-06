"""Exercise 04: Special Token Handler (Easy, numpy)

LLMs use special tokens to mark structure in their input:
    - BOS (Beginning of Sequence): signals the start of a new sequence
    - EOS (End of Sequence): signals the end of a sequence
    - PAD (Padding): fills sequences to uniform length in a batch

Proper handling of these tokens is critical for correct training and inference:
    - BOS/EOS must be added correctly before/after content tokens
    - Batch padding can be left-sided (common for decoder-only inference) or
      right-sided (common for training)
    - Attention masks must mark which positions are real tokens (1) vs padding (0)

Implement:
    - encode_with_specials: add BOS/EOS tokens to a sequence
    - pad_batch: pad a list of variable-length sequences to uniform length
    - create_attention_mask: generate 0/1 mask marking real tokens
"""

import numpy as np


def encode_with_specials(
    token_ids: list[int],
    bos_id: int,
    eos_id: int,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[int]:
    """Add BOS and/or EOS special tokens to a sequence.

    Args:
        token_ids: list of integer token IDs (the content)
        bos_id: token ID for beginning-of-sequence
        eos_id: token ID for end-of-sequence
        add_bos: whether to prepend BOS
        add_eos: whether to append EOS

    Returns:
        New list with BOS/EOS added as specified.
    """
    # TODO: prepend bos_id if add_bos, append eos_id if add_eos
    raise NotImplementedError("Implement encode_with_specials")


def pad_batch(
    sequences: list[list[int]],
    pad_id: int,
    padding_side: str = "right",
    max_len: int | None = None,
) -> np.ndarray:
    """Pad a batch of variable-length sequences to uniform length.

    Args:
        sequences: list of token ID lists, possibly different lengths
        pad_id: token ID to use for padding
        padding_side: 'left' or 'right' - which side to add padding
        max_len: target length. If None, use the length of the longest sequence.

    Returns:
        2D numpy array of shape (batch_size, max_len), dtype int64.
    """
    # TODO:
    # 1. Determine max_len (from sequences or argument)
    # 2. For each sequence, truncate if longer than max_len
    # 3. Pad to max_len on the specified side
    raise NotImplementedError("Implement pad_batch")


def create_attention_mask(padded_ids: np.ndarray, pad_id: int) -> np.ndarray:
    """Create an attention mask from padded token IDs.

    Args:
        padded_ids: 2D array of shape (batch_size, seq_len) with padding
        pad_id: the token ID used for padding

    Returns:
        2D numpy array of same shape, dtype int64.
        1 where token is real, 0 where token is padding.
    """
    # TODO: return 1 where padded_ids != pad_id, 0 otherwise
    raise NotImplementedError("Implement create_attention_mask")
