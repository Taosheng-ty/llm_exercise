"""
Exercise 02: Generalized Advantage Estimation (GAE) in PyTorch
===============================================================

Implement vectorized GAE using PyTorch tensors (not numpy).

Given:
- rewards: (batch, seq_len) per-token rewards
- values:  (batch, seq_len) value estimates V(s_t)
- gamma:   discount factor (e.g., 1.0)
- lambd:   GAE lambda (e.g., 0.95)

Compute advantages and returns using GAE:
1. delta_t = rewards_t + gamma * V(s_{t+1}) - V(s_t)   [V(s_T) = 0 for terminal]
2. A_t = delta_t + gamma * lambd * A_{t+1}              [A_T = delta_T]
3. returns_t = A_t + V(s_t)

The computation should handle batched inputs (batch, seq_len) and process
all sequences in the batch simultaneously.

Reference: slime/utils/ppo_utils.py :: vanilla_gae() and get_advantages_and_returns()
"""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lambd: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (batch, seq_len) per-token rewards.
        values:  (batch, seq_len) value estimates.
        gamma:   discount factor.
        lambd:   GAE lambda.

    Returns:
        advantages: (batch, seq_len) GAE advantages.
        returns:    (batch, seq_len) = advantages + values.
    """
    # TODO: Implement vectorized GAE
    # Hint 1: Compute TD residuals delta_t = r_t + gamma * V_{t+1} - V_t
    #          where V_T = 0 (terminal value)
    # Hint 2: Walk backwards: A_t = delta_t + gamma * lambd * A_{t+1}
    # Hint 3: returns = advantages + values
    raise NotImplementedError("Implement compute_gae")
