"""
Solution for Exercise 01: Generalized Advantage Estimation (GAE)
"""

import numpy as np


def vanilla_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lambd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE)."""
    B, T = rewards.shape

    lastgaelam = np.zeros(B)
    adv_reversed = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_reversed.append(lastgaelam.copy())

    # Reverse to get correct time order and stack into (B, T)
    advantages = np.stack(adv_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns
