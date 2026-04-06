"""
Solution for Exercise 03: Identity Preference Optimization (IPO) Loss
"""

import torch


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute IPO loss.

    IPO optimizes: loss = ((log_ratio_chosen - log_ratio_rejected) - 1/(2*beta))^2
    This squared loss targets a specific margin rather than maximizing it.
    """
    log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

    margin = log_ratio_chosen - log_ratio_rejected
    target = 1.0 / (2.0 * beta)

    loss = ((margin - target) ** 2).mean()

    metrics = {
        "log_ratio_chosen": log_ratio_chosen,
        "log_ratio_rejected": log_ratio_rejected,
        "margin": margin,
        "target": target,
    }
    return loss, metrics
