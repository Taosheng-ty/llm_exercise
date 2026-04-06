"""
Exercise 01: Bradley-Terry Reward Model

Difficulty: Hard
Framework: PyTorch

Background:
    The Bradley-Terry model is the standard framework for learning reward models
    from human preference data. Given a pair of responses (chosen, rejected) to
    a prompt, the probability that the chosen response is preferred is modeled as:

        P(chosen > rejected) = sigmoid(r_chosen - r_rejected)

    where r_chosen and r_rejected are scalar reward scores from the model.

    The training loss is the negative log-likelihood:

        loss = -log(sigmoid(r_chosen - r_rejected))

    This is equivalent to binary cross-entropy where the "label" is always 1
    (the chosen response should always score higher).

Implement:
    reward_model_loss(chosen_rewards, rejected_rewards) -> (loss, accuracy)
        Compute the Bradley-Terry pairwise ranking loss and classification accuracy.

    compute_reward_margins(chosen_rewards, rejected_rewards) -> margins
        Compute the margin r_chosen - r_rejected for each pair.

Args for reward_model_loss:
    chosen_rewards: (B,) scalar rewards for chosen responses
    rejected_rewards: (B,) scalar rewards for rejected responses

Returns:
    loss: scalar tensor, mean of -log(sigmoid(r_chosen - r_rejected))
    accuracy: float, fraction of pairs where r_chosen > r_rejected

Args for compute_reward_margins:
    chosen_rewards: (B,) scalar rewards for chosen responses
    rejected_rewards: (B,) scalar rewards for rejected responses

Returns:
    margins: (B,) tensor of r_chosen - r_rejected
"""

import torch
import torch.nn.functional as F


def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Compute Bradley-Terry pairwise ranking loss and accuracy.

    Args:
        chosen_rewards: (B,) reward scores for preferred responses
        rejected_rewards: (B,) reward scores for rejected responses

    Returns:
        loss: scalar tensor
        accuracy: float, fraction where chosen > rejected
    """
    # TODO: Implement Bradley-Terry loss
    # Hint 1: Compute the margin: r_chosen - r_rejected
    # Hint 2: loss = -F.logsigmoid(margin).mean()
    # Hint 3: accuracy = (margin > 0).float().mean().item()
    raise NotImplementedError("Implement reward_model_loss")


def compute_reward_margins(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Compute reward margins between chosen and rejected responses.

    Args:
        chosen_rewards: (B,) reward scores for preferred responses
        rejected_rewards: (B,) reward scores for rejected responses

    Returns:
        margins: (B,) tensor of (chosen - rejected) margins
    """
    # TODO: Implement margin computation
    raise NotImplementedError("Implement compute_reward_margins")
