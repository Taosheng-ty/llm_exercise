"""
Solution 02: Generalized Advantage Estimation (GAE) in PyTorch
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
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    # Step 1: Compute TD residuals
    # delta_t = r_t + gamma * V_{t+1} - V_t, with V_T = 0
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Step 2: Backward pass to accumulate GAE
    # A_t = delta_t + gamma * lambd * A_{t+1}
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    advantages_reversed = []

    for t in reversed(range(T)):
        lastgaelam = deltas[:, t] + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)

    # Step 3: Reverse to get correct time order and stack
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    # Step 4: Returns = advantages + values
    returns = advantages + values

    return advantages, returns
