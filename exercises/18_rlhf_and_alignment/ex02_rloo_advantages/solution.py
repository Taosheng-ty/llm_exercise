"""
Solution for Exercise 02: REINFORCE Leave-One-Out (RLOO) Advantages
"""

import numpy as np


def rloo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Compute REINFORCE Leave-One-Out advantages.

    For each completion i, advantage_i = r_i - mean(r_j for j != i).
    Vectorized: baseline_i = (total - r_i) / (K - 1), advantage_i = r_i - baseline_i.
    """
    num_prompts, K = rewards.shape

    if K == 1:
        return np.zeros_like(rewards)

    # Sum of all rewards per prompt
    total = rewards.sum(axis=1, keepdims=True)  # (num_prompts, 1)

    # Leave-one-out baseline for each completion
    loo_mean = (total - rewards) / (K - 1)  # (num_prompts, K)

    # Advantage = reward - leave-one-out baseline
    advantages = rewards - loo_mean

    return advantages
