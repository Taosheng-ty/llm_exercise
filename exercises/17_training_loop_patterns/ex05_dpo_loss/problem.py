"""
Exercise 05: Direct Preference Optimization (DPO) Loss

Difficulty: Hard
Framework: PyTorch

Background:
    DPO aligns language models with human preferences without explicit reward
    modeling. Given pairs of (chosen, rejected) responses, DPO optimizes:

    loss = -log(sigmoid(beta * ((log_pi(chosen) - log_ref(chosen))
                                - (log_pi(rejected) - log_ref(rejected)))))

    Where:
    - log_pi(x) = sum of log probs of x under the policy model
    - log_ref(x) = sum of log probs of x under the reference model
    - beta controls the strength of the KL constraint

    The intuition: increase the gap between chosen and rejected responses
    relative to the reference model.

Implement:
    dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta) -> (loss, metrics)

    Also implement:
    compute_log_probs(logits, labels, mask) -> per-sample sum of log probs

Args for dpo_loss:
    policy_chosen_logps: (B,) log probs of chosen under policy
    policy_rejected_logps: (B,) log probs of rejected under policy
    ref_chosen_logps: (B,) log probs of chosen under reference
    ref_rejected_logps: (B,) log probs of rejected under reference
    beta: float - temperature parameter

Returns:
    loss: scalar tensor
    metrics: dict with keys 'chosen_rewards', 'rejected_rewards', 'accuracy'
        - chosen_rewards: (B,) = beta * (policy_chosen_logps - ref_chosen_logps)
        - rejected_rewards: (B,) = beta * (policy_rejected_logps - ref_rejected_logps)
        - accuracy: float, fraction where chosen_reward > rejected_reward
"""

import torch
import torch.nn.functional as F


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample sum of log probabilities.

    Args:
        logits: (B, T, V) model logits
        labels: (B, T) token ids
        mask: (B, T) binary mask for which tokens to include

    Returns:
        (B,) sum of log probs for each sample (only masked positions).
    """
    # TODO: Implement log prob computation
    # Hint 1: Shift logits[:, :-1] and labels[:, 1:] (causal LM)
    # Hint 2: Use F.log_softmax then gather the label indices
    # Hint 3: Multiply by shifted mask, sum over sequence dim
    raise NotImplementedError("Implement compute_log_probs")


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute DPO loss.

    Returns:
        (loss, metrics_dict)
    """
    # TODO: Implement DPO loss
    # Hint 1: chosen_rewards = beta * (policy_chosen - ref_chosen)
    # Hint 2: rejected_rewards = beta * (policy_rejected - ref_rejected)
    # Hint 3: logits = chosen_rewards - rejected_rewards
    # Hint 4: loss = -F.logsigmoid(logits).mean()
    # Hint 5: accuracy = (chosen_rewards > rejected_rewards).float().mean()
    raise NotImplementedError("Implement dpo_loss")
