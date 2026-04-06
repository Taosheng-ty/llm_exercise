"""
Exercise 04: GRPO (Group Relative Policy Optimization) Advantage Estimation

Difficulty: Easy

Background:
    GRPO generates multiple completions (a "group") for the same prompt, then
    normalizes the rewards within the group to compute advantages:

        advantage_i = (reward_i - mean(rewards)) / std(rewards)

    Use population standard deviation (ddof=0) when computing std.

    This removes the need for a learned value function -- the group's mean
    reward acts as a baseline, and dividing by std normalizes the scale.

    Each token in a response gets the same advantage (the sequence-level
    normalized reward).

    Edge case: if all rewards in a group are identical (std=0), advantages
    should be 0 to avoid division by zero.

    Reference: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
    in Open Language Models" (Shao et al., 2024)
    Also see: slime/utils/ppo_utils.py get_grpo_returns()

Args:
    rewards:          numpy array of shape (G,) - scalar reward for each completion in the group
    response_lengths: list of int of length G - number of tokens in each response

Returns:
    advantages: list of G numpy arrays, where advantages[i] has shape (response_lengths[i],)
                and every element equals the normalized reward for that response
"""

import numpy as np


def compute_grpo_advantages(
    rewards: np.ndarray,
    response_lengths: list[int],
) -> list[np.ndarray]:
    """Compute GRPO group-normalized advantages.

    See module docstring for full details.
    """
    # TODO: Implement GRPO advantage computation
    # Hint 1: Compute mean and std of the reward group
    # Hint 2: Handle the std=0 edge case (all rewards identical)
    # Hint 3: Broadcast each normalized reward to all tokens in the response
    raise NotImplementedError("Implement compute_grpo_advantages")
