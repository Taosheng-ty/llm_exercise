"""
Solution for Exercise 04: GRPO Advantage Estimation
"""

import numpy as np


def compute_grpo_advantages(
    rewards: np.ndarray,
    response_lengths: list[int],
) -> list[np.ndarray]:
    """Compute GRPO group-normalized advantages."""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if std_reward < 1e-8:
        # All rewards identical -- advantages are zero
        normalized = np.zeros_like(rewards)
    else:
        normalized = (rewards - mean_reward) / std_reward

    advantages = []
    for i, length in enumerate(response_lengths):
        advantages.append(np.full(length, normalized[i]))

    return advantages
