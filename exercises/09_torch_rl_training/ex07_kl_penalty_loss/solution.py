"""
Solution 07: KL Divergence Penalty Loss
"""

import torch


def compute_approx_kl(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    kl_type: str = "k3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute approximate KL divergence between policy and reference.

    Args:
        policy_log_probs: (batch, seq_len) current policy log-probs.
        ref_log_probs:    (batch, seq_len) reference policy log-probs.
        loss_mask:        (batch, seq_len) binary mask.
        kl_type:          one of "k1", "k2", "k3".

    Returns:
        per_token_kl: (batch, seq_len) per-token KL estimates.
        mean_kl:      scalar, masked mean KL.
    """
    log_ratio = policy_log_probs.float() - ref_log_probs.float()

    if kl_type == "k1":
        per_token_kl = log_ratio
    elif kl_type == "k2":
        per_token_kl = log_ratio ** 2 / 2.0
    elif kl_type == "k3":
        # Non-negative KL approximation: exp(-log_ratio) - 1 - (-log_ratio)
        neg_log_ratio = -log_ratio
        per_token_kl = neg_log_ratio.exp() - 1 - neg_log_ratio
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    # Masked mean
    num_valid = torch.clamp_min(loss_mask.sum(), 1.0)
    mean_kl = (per_token_kl * loss_mask).sum() / num_valid

    return per_token_kl, mean_kl


def compute_loss_with_kl_penalty(
    policy_loss: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    beta: float = 0.1,
    kl_type: str = "k3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add KL penalty to policy loss.

    Args:
        policy_loss:      scalar policy gradient loss.
        policy_log_probs: (batch, seq_len) current policy log-probs.
        ref_log_probs:    (batch, seq_len) reference log-probs.
        loss_mask:        (batch, seq_len) binary mask.
        beta:             KL penalty coefficient.
        kl_type:          KL estimator type.

    Returns:
        total_loss: scalar = policy_loss + beta * mean_kl.
        mean_kl:    scalar, for logging.
    """
    _, mean_kl = compute_approx_kl(policy_log_probs, ref_log_probs, loss_mask, kl_type)
    total_loss = policy_loss + beta * mean_kl
    return total_loss, mean_kl
