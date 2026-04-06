"""
Solution for Exercise 05: Direct Preference Optimization (DPO) Loss
"""

import torch
import torch.nn.functional as F


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample sum of log probabilities."""
    # Shift for causal LM: logits[:, :-1] predict labels[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs at label positions
    per_token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Apply mask and sum over sequence
    return (per_token_logps * shift_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute DPO loss."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()

    accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "accuracy": accuracy,
    }
    return loss, metrics
