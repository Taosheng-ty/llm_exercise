"""
Exercise 01: PPO Clipped Policy Gradient Loss in PyTorch
=========================================================

Implement the PPO clipped surrogate objective as a differentiable PyTorch function.

In RLHF/PPO training of LLMs, the policy gradient loss drives the model to increase
probability of tokens in high-reward responses while decreasing probability of tokens
in low-reward responses. The clipping mechanism prevents destructive updates that could
collapse the model's language capabilities.

Given:
- old_log_probs: (batch, seq_len) log-probs under the old policy
- new_log_probs: (batch, seq_len) log-probs under the current policy (requires grad)
- advantages:    (batch, seq_len) advantage estimates
- loss_mask:     (batch, seq_len) binary mask (1 = valid token, 0 = padding)
- eps_clip:      clipping threshold (default 0.2)

Compute:
1. ratio = exp(new_log_probs - old_log_probs)
2. surr1 = -ratio * advantages
3. surr2 = -clamp(ratio, 1-eps, 1+eps) * advantages
4. per-token loss = max(surr1, surr2)   [pessimistic clipping]
5. Return the masked mean loss (mean over valid tokens) AND the clip fraction
   (fraction of valid tokens where surr2 > surr1).

The loss must be differentiable: loss.backward() must produce gradients on new_log_probs.

Reference: slime/utils/ppo_utils.py :: compute_policy_loss()
           slime/backends/megatron_utils/loss.py :: policy_loss_function()
"""

import torch


def compute_ppo_clipped_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    eps_clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO clipped policy gradient loss.

    Args:
        old_log_probs: (batch, seq_len) old policy log-probs, detached.
        new_log_probs: (batch, seq_len) current policy log-probs, requires grad.
        advantages:    (batch, seq_len) advantage values.
        loss_mask:     (batch, seq_len) binary mask for valid tokens.
        eps_clip:      PPO clipping epsilon.

    Returns:
        loss: scalar, masked mean of per-token clipped loss.
        clip_frac: scalar, fraction of valid tokens that were clipped.
    """
    # TODO: Implement PPO clipped loss
    # Hint 1: ratio = exp(new_log_probs - old_log_probs)
    # Hint 2: surr1 = -ratio * advantages
    # Hint 3: surr2 = -torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
    # Hint 4: per_token_loss = torch.maximum(surr1, surr2)
    # Hint 5: clip_frac = (surr2 > surr1).float() masked mean
    raise NotImplementedError("Implement compute_ppo_clipped_loss")
