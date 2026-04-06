"""
Solution for Exercise 05: REINFORCE with Baseline (Discounted Returns)
"""

import numpy as np


def compute_discounted_returns(
    token_rewards: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns G_t = r_t + gamma * G_{t+1} for each timestep."""
    T = len(token_rewards)
    returns = np.zeros(T)
    running_return = 0.0

    for t in reversed(range(T)):
        running_return = token_rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def reinforce_with_baseline(
    rewards_list: list[np.ndarray],
    gamma: float,
) -> list[np.ndarray]:
    """Compute REINFORCE advantages with a group-mean baseline."""
    # Compute discounted returns for each sequence
    returns_list = [compute_discounted_returns(r, gamma) for r in rewards_list]

    # Baseline = mean of G_0 across sequences
    baseline = np.mean([ret[0] for ret in returns_list])

    # Subtract baseline from all returns to get advantages
    advantages = [ret - baseline for ret in returns_list]

    return advantages
