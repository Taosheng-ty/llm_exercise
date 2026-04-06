"""
Exercise 05: REINFORCE with Baseline (Discounted Returns)

Difficulty: Medium

Background:
    REINFORCE++ computes token-level discounted returns for policy gradient
    training. The key steps are:

    1. Assign token-level rewards: most tokens get 0 reward, but the LAST
       token in the response gets the full sequence reward.
    2. Compute discounted returns backwards:
       G_t = r_t + gamma * G_{t+1}
    3. Optionally subtract a baseline (mean return across a group) to
       reduce variance.

    This exercise implements a simplified version:
    - Given a sequence of token-level rewards (where the final token has the
      main reward), compute discounted returns G_t for each position.
    - Given a group of such return sequences, subtract the group-mean return
      from each to produce advantages (the baseline subtraction).

    Reference: "REINFORCE++: A Simple and Efficient Approach for Aligning
    Large Language Models" (Hu, 2025)
    Also see: slime/utils/ppo_utils.py get_reinforce_plus_plus_returns()

Functions to implement:
    1. compute_discounted_returns(token_rewards, gamma)
       - token_rewards: 1D array of per-token rewards
       - gamma: discount factor
       - Returns: 1D array of discounted returns G_t

    2. reinforce_with_baseline(rewards_list, gamma)
       - rewards_list: list of 1D arrays (one per sequence in the group)
       - gamma: discount factor
       - Returns: list of 1D advantage arrays (returns minus group baseline)
"""

import numpy as np


def compute_discounted_returns(
    token_rewards: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns G_t = r_t + gamma * G_{t+1} for each timestep.

    Args:
        token_rewards: 1D array of per-token rewards.
        gamma: Discount factor.

    Returns:
        1D array of discounted returns, same shape as token_rewards.
    """
    # TODO: Implement backward computation of discounted returns
    # Hint: Iterate backwards, accumulating: G_t = r_t + gamma * G_{t+1}
    raise NotImplementedError("Implement compute_discounted_returns")


def reinforce_with_baseline(
    rewards_list: list[np.ndarray],
    gamma: float,
) -> list[np.ndarray]:
    """Compute REINFORCE advantages with a group-mean baseline.

    For each sequence, compute discounted returns, then subtract the group-mean
    baseline. The baseline is the mean of G_0 values across all sequences in the
    group, where G_0 is the discounted return at the first timestep (index 0)
    of each sequence.

    Args:
        rewards_list: List of 1D token-reward arrays (one per sequence).
        gamma: Discount factor.

    Returns:
        List of 1D advantage arrays (returns - baseline), one per sequence.
    """
    # TODO: Implement REINFORCE with baseline
    # Hint 1: Compute discounted returns for each sequence
    # Hint 2: Compute baseline as mean of G_0 values across all sequences
    # Hint 3: Subtract baseline from all returns
    raise NotImplementedError("Implement reinforce_with_baseline")
