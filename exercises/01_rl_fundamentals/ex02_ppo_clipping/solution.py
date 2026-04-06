"""
Solution for Exercise 02: PPO Clipped Surrogate Objective
"""

import numpy as np


def compute_policy_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    eps_clip: float,
) -> tuple[np.ndarray, float]:
    """Compute PPO clipped surrogate policy loss."""
    ratio = np.exp(log_probs_new - log_probs_old)

    # Unclipped objective
    pg_losses1 = -ratio * advantages

    # Clipped objective
    clipped_ratio = np.clip(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_losses2 = -clipped_ratio * advantages

    # Take the pessimistic (maximum) bound
    policy_loss = np.maximum(pg_losses1, pg_losses2)

    # Fraction of samples where clipping was active
    clip_fraction = float(np.mean(pg_losses2 > pg_losses1))

    return policy_loss, clip_fraction
