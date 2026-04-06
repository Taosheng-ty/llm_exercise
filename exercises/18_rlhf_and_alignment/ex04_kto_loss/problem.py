"""
Exercise 04: Kahneman-Tversky Optimization (KTO) Loss

Difficulty: Medium
Framework: PyTorch

Background:
    KTO (Ethayarajh et al., 2024) enables preference optimization with *unpaired*
    binary feedback (thumbs up / thumbs down) rather than requiring paired
    comparisons (chosen vs rejected).

    Given a sample with log-ratio = log(pi(y|x)) - log(pi_ref(y|x)):

    For desirable samples (thumbs up):
        loss = 1 - sigmoid(beta * (log_ratio - kl_estimate))

    For undesirable samples (thumbs down):
        loss = 1 - sigmoid(beta * (kl_estimate - log_ratio))

    Where kl_estimate is a running estimate of KL(pi || pi_ref) computed
    externally (provided as input).

    Intuition: desirable samples should have high log-ratio (policy favors them
    more than reference), while undesirable samples should have low log-ratio.

Implement:
    kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate, beta) -> (loss, metrics)

Args:
    policy_logps: (B,) log probs under policy
    ref_logps: (B,) log probs under reference
    is_desirable: (B,) boolean tensor, True for desirable (positive) feedback
    kl_estimate: scalar float, estimate of KL(pi || pi_ref)
    beta: float - temperature parameter

Returns:
    loss: scalar tensor, mean loss over the batch
    metrics: dict with keys:
        'log_ratios': (B,) = policy_logps - ref_logps
        'desirable_loss': scalar, mean loss over desirable samples (0.0 if none)
        'undesirable_loss': scalar, mean loss over undesirable samples (0.0 if none)
"""

import torch


def kto_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    is_desirable: torch.Tensor,
    kl_estimate: float,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute KTO loss for unpaired binary preference data.

    Returns:
        (loss, metrics_dict)
    """
    # TODO: Implement KTO loss
    # Hint 1: log_ratios = policy_logps - ref_logps
    # Hint 2: For desirable: loss_d = 1 - sigmoid(beta * (log_ratios - kl_estimate))
    # Hint 3: For undesirable: loss_u = 1 - sigmoid(beta * (kl_estimate - log_ratios))
    # Hint 4: Combine: loss = mean of all individual losses
    # Hint 5: Handle case where all samples are desirable or all undesirable
    raise NotImplementedError("Implement kto_loss")
