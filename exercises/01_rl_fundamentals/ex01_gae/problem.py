"""
Exercise 01: Generalized Advantage Estimation (GAE)

Difficulty: Medium

Background:
    In Proximal Policy Optimization (PPO), Generalized Advantage Estimation (GAE)
    is used to compute advantage estimates that balance bias and variance.

    The key idea:
    1. Compute TD residuals: delta_t = r_t + gamma * V(t+1) - V(t)
       (where V(T) = 0 for the terminal step)
    2. Accumulate advantages backwards:
       A_t = delta_t + gamma * lambda * A_{t+1}
    3. Returns = Advantages + Values

    Reference: "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation" (Schulman et al., 2016)
    Also see: slime/utils/ppo_utils.py vanilla_gae()

Args:
    rewards: numpy array of shape (batch_size, seq_len) - rewards at each timestep
    values:  numpy array of shape (batch_size, seq_len) - value estimates at each timestep
    gamma:   float - discount factor (typically 0.99)
    lambd:   float - GAE lambda for bias-variance tradeoff (typically 0.95)

Returns:
    advantages: numpy array of shape (batch_size, seq_len)
    returns:    numpy array of shape (batch_size, seq_len), where returns = advantages + values
"""

import numpy as np


def vanilla_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lambd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    See module docstring for full details.
    """
    # TODO: Implement GAE
    # Hint 1: Iterate backwards through timesteps (reversed(range(T)))
    # Hint 2: For the last timestep, next_value = 0.0
    # Hint 3: delta = reward + gamma * next_value - current_value
    # Hint 4: Accumulate: lastgaelam = delta + gamma * lambd * lastgaelam
    raise NotImplementedError("Implement vanilla_gae")
