"""
Solution 08: GRPO Loss Function in PyTorch
"""

import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-normalized advantages from per-response rewards.

    Args:
        rewards: (num_prompts, group_size) scalar rewards.
        eps: small constant for numerical stability.

    Returns:
        advantages: (num_prompts, group_size) normalized advantages.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean) / (std + eps)
    return advantages


def compute_grpo_loss(
    new_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    ref_log_probs: list[torch.Tensor],
    rewards: torch.Tensor,
    loss_masks: list[torch.Tensor],
    group_size: int,
    eps_clip: float = 0.2,
    kl_coef: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute full GRPO loss.

    Args:
        new_log_probs:  list of (seq_len_i,) tensors, current policy.
        old_log_probs:  list of (seq_len_i,) tensors, old policy.
        ref_log_probs:  list of (seq_len_i,) tensors, reference policy.
        rewards:        (num_prompts, group_size) rewards.
        loss_masks:     list of (seq_len_i,) binary masks.
        group_size:     number of responses per prompt.
        eps_clip:       PPO clipping epsilon.
        kl_coef:        KL penalty coefficient.

    Returns:
        loss: scalar, total GRPO loss (differentiable).
        metrics: dict with "pg_loss", "kl_loss", "mean_advantage", "clip_frac".
    """
    num_prompts = rewards.shape[0]
    assert len(new_log_probs) == num_prompts * group_size

    # Step 1: Compute group-normalized advantages
    advantages = compute_group_advantages(rewards)  # (num_prompts, group_size)

    # Step 2: Broadcast advantages to per-token and compute PPO loss
    all_pg_losses = []
    all_clip_flags = []
    all_kl = []
    all_masks = []

    for prompt_idx in range(num_prompts):
        for resp_idx in range(group_size):
            flat_idx = prompt_idx * group_size + resp_idx
            adv_scalar = advantages[prompt_idx, resp_idx]

            new_lp = new_log_probs[flat_idx]
            old_lp = old_log_probs[flat_idx]
            ref_lp = ref_log_probs[flat_idx]
            mask = loss_masks[flat_idx]

            # Broadcast advantage to all tokens of this response
            token_advantages = adv_scalar * torch.ones_like(new_lp)

            # PPO-clipped loss: ppo_kl = old_lp - new_lp, ratio = exp(-ppo_kl)
            ppo_kl = old_lp - new_lp
            ratio = (-ppo_kl).exp()

            surr1 = -ratio * token_advantages
            surr2 = -ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * token_advantages
            pg_loss = torch.maximum(surr1, surr2)
            clip_flag = (surr2 > surr1).float()

            # KL penalty: k3 estimator (non-negative)
            log_ratio = new_lp.float() - ref_lp.float()
            neg_log_ratio = -log_ratio
            kl = neg_log_ratio.exp() - 1 - neg_log_ratio

            all_pg_losses.append(pg_loss)
            all_clip_flags.append(clip_flag)
            all_kl.append(kl)
            all_masks.append(mask)

    # Concatenate all tokens across all responses
    all_pg_losses = torch.cat(all_pg_losses, dim=0)
    all_clip_flags = torch.cat(all_clip_flags, dim=0)
    all_kl = torch.cat(all_kl, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Step 3: Masked mean
    num_valid = torch.clamp_min(all_masks.sum(), 1.0)
    pg_loss = (all_pg_losses * all_masks).sum() / num_valid
    kl_loss = (all_kl * all_masks).sum() / num_valid
    clip_frac = (all_clip_flags * all_masks).sum() / num_valid

    # Step 4: Total loss
    loss = pg_loss + kl_coef * kl_loss

    metrics = {
        "pg_loss": pg_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mean_advantage": advantages.mean().detach(),
        "clip_frac": clip_frac.detach(),
    }

    return loss, metrics
