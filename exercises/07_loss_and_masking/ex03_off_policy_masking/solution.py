"""
Solution for Exercise 03: Off-Policy Sequence Masking (OPSM)
"""
import numpy as np


def compute_opsm_mask(
    seq_kl: np.ndarray,
    advantages: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Compute Off-Policy Sequence Masking (OPSM).

    Args:
        seq_kl: Per-sequence KL divergence values, shape (num_sequences,).
        advantages: Per-sequence advantage values, shape (num_sequences,).
        delta: KL divergence threshold. Sequences with KL > delta and
               advantage < 0 are masked out.

    Returns:
        Tuple of:
          - mask: Binary array of shape (num_sequences,). 1 = keep, 0 = masked.
          - clip_fraction: Fraction of sequences that were masked out (float).
    """
    # Identify sequences to mask: advantage < 0 AND kl > delta
    should_mask = (advantages < 0) & (seq_kl > delta)

    # Mask: 1 = keep, 0 = masked out
    mask = (~should_mask).astype(float)

    # Fraction of masked sequences
    num_sequences = len(advantages)
    if num_sequences == 0:
        clip_fraction = 0.0
    else:
        clip_fraction = float(should_mask.sum()) / num_sequences

    return mask, clip_fraction
