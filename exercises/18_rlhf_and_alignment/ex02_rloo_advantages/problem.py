"""
Exercise 02: REINFORCE Leave-One-Out (RLOO) Advantages

Difficulty: Medium
Framework: numpy

Background:
    REINFORCE Leave-One-Out is a variance reduction technique for policy gradient
    methods. Instead of using a learned value function as baseline, RLOO uses the
    mean reward of the *other* completions for the same prompt as the baseline.

    Given K completions per prompt with rewards r_1, ..., r_K, the RLOO advantage
    for completion i is:

        advantage_i = r_i - mean(r_j for j != i)

    The leave-one-out mean for completion i is:
        baseline_i = (sum(r_1, ..., r_K) - r_i) / (K - 1)

    This provides a lower-variance estimate than using a single baseline because
    each completion gets a customized baseline from its peers.

Implement:
    rloo_advantages(rewards) -> advantages
        Compute RLOO advantages for each completion.

Args:
    rewards: np.ndarray of shape (num_prompts, K) where K is completions per prompt

Returns:
    advantages: np.ndarray of shape (num_prompts, K), the RLOO advantages

Edge cases:
    - When K=1, there are no other completions, so advantage should be 0.
"""

import numpy as np


def rloo_advantages(rewards: np.ndarray) -> np.ndarray:
    """Compute REINFORCE Leave-One-Out advantages.

    For each completion i of a prompt, the advantage is:
        advantage_i = reward_i - mean(reward_j for j != i)

    Args:
        rewards: (num_prompts, K) reward values, K completions per prompt

    Returns:
        advantages: (num_prompts, K) RLOO advantage estimates
    """
    # TODO: Implement RLOO advantages
    # Hint 1: Compute sum of all rewards per prompt: total = rewards.sum(axis=1, keepdims=True)
    # Hint 2: Leave-one-out mean for i: (total - rewards[:, i]) / (K - 1)
    # Hint 3: Advantage_i = rewards[:, i] - leave_one_out_mean_i
    # Hint 4: Handle K=1 edge case (return zeros)
    raise NotImplementedError("Implement rloo_advantages")
