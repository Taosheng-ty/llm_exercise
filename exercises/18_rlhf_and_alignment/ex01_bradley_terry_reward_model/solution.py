"""
Solution for Exercise 01: Bradley-Terry Reward Model
"""

import torch
import torch.nn.functional as F


def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compute Bradley-Terry pairwise ranking loss and accuracy.

    The Bradley-Terry model defines P(chosen > rejected) = sigmoid(r_c - r_r).
    The loss is the negative log-likelihood: -log(sigmoid(r_c - r_r)).
    """
    margins = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margins).mean()
    accuracy = (margins > 0).float().mean().item()
    return loss, accuracy


def compute_reward_margins(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Compute reward margins between chosen and rejected responses."""
    return chosen_rewards - rejected_rewards
