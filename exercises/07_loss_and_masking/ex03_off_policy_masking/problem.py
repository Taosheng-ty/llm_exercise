"""
Exercise 03: Off-Policy Sequence Masking (OPSM)

During PPO training, the policy changes between rollout and training. If the
policy has drifted too far (high KL divergence) for sequences with negative
advantage, we should mask those sequences out to avoid harmful updates.

OPSM rule: mask a sequence (set mask = 0) when BOTH conditions hold:
  - The sequence has negative advantage (advantage < 0)
  - The per-sequence KL divergence exceeds a threshold delta (kl > delta)

Your task:
  1. For each sequence, check both conditions.
  2. Return a binary mask (1 = keep, 0 = masked out).
  3. Return the fraction of sequences that were masked.

Reference: slime's compute_opsm_mask() in ppo_utils.py.

Difficulty: Easy
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
    # TODO: Implement OPSM
    #   1. Identify sequences where advantage < 0 AND kl > delta.
    #   2. Set those to 0, everything else to 1.
    #   3. Compute the fraction of masked sequences.
    raise NotImplementedError
