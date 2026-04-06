"""
Solution for Exercise 04: Kahneman-Tversky Optimization (KTO) Loss
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

    Desirable: loss = 1 - sigmoid(beta * (log_ratio - kl_estimate))
    Undesirable: loss = 1 - sigmoid(beta * (kl_estimate - log_ratio))
    """
    log_ratios = policy_logps - ref_logps

    # Compute per-sample losses
    desirable_mask = is_desirable.bool()
    undesirable_mask = ~desirable_mask

    # Initialize losses as zeros
    per_sample_loss = torch.zeros_like(log_ratios)

    if desirable_mask.any():
        per_sample_loss[desirable_mask] = 1.0 - torch.sigmoid(
            beta * (log_ratios[desirable_mask] - kl_estimate)
        )

    if undesirable_mask.any():
        per_sample_loss[undesirable_mask] = 1.0 - torch.sigmoid(
            beta * (kl_estimate - log_ratios[undesirable_mask])
        )

    loss = per_sample_loss.mean()

    # Compute per-type losses for metrics
    desirable_loss = (
        per_sample_loss[desirable_mask].mean().item()
        if desirable_mask.any()
        else 0.0
    )
    undesirable_loss = (
        per_sample_loss[undesirable_mask].mean().item()
        if undesirable_mask.any()
        else 0.0
    )

    metrics = {
        "log_ratios": log_ratios,
        "desirable_loss": desirable_loss,
        "undesirable_loss": undesirable_loss,
    }
    return loss, metrics
