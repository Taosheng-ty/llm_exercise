"""
Exercise 07: KL Divergence Penalty Loss
=========================================

Implement approximate KL divergence computation and add it as a penalty
to the policy loss, supporting multiple KL estimator types.

The KL penalty is a cornerstone of RLHF — it prevents the policy from drifting too far
from the pretrained reference model during RL fine-tuning. Without this constraint, the
model may "reward hack" by generating degenerate but high-reward outputs that lose the
general language capabilities learned during pretraining.

Given:
- policy_log_probs: (batch, seq_len) log-probs from current policy
- ref_log_probs:    (batch, seq_len) log-probs from reference policy
- loss_mask:        (batch, seq_len) binary mask
- kl_type:          estimator type: "k1", "k2", or "k3"

KL estimator types (Schulman blog: http://joschu.net/blog/kl-approx.html):
- k1: KL ~ log_ratio                                (simplest, biased)
- k2: KL ~ log_ratio^2 / 2                          (chi-squared)
- k3: KL ~ exp(-log_ratio) - 1 + log_ratio          (non-negative, low variance)

Then combine: total_loss = policy_loss + beta * mean_kl

Reference: slime/utils/ppo_utils.py :: compute_approx_kl()
           slime/backends/megatron_utils/loss.py :: policy_loss_function() (use_kl_loss)
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
    # TODO: Implement KL computation
    # Hint: log_ratio = policy_log_probs - ref_log_probs
    # k1: kl = log_ratio
    # k2: kl = log_ratio^2 / 2
    # k3: kl = exp(-log_ratio) - 1 - (-log_ratio)
    raise NotImplementedError("Implement compute_approx_kl")


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
    # TODO: Combine policy loss with KL penalty
    raise NotImplementedError("Implement compute_loss_with_kl_penalty")
