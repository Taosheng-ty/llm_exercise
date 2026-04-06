"""
Exercise 08: GRPO Loss Function in PyTorch (Hard)
===================================================

Implement the full Group Relative Policy Optimization (GRPO) loss function.

GRPO is used in training LLMs where multiple responses are generated per prompt.
The advantage for each response is computed relative to the group (per-prompt).

Given:
- new_log_probs:  list of G tensors, each (seq_len_g,) -- current policy log-probs
- old_log_probs:  list of G tensors, each (seq_len_g,) -- old policy log-probs
- ref_log_probs:  list of G tensors, each (seq_len_g,) -- reference policy log-probs
- rewards:        (num_prompts, G) scalar rewards for each response
- loss_masks:     list of G tensors, each (seq_len_g,) -- binary masks
- group_size:     G, number of responses per prompt
- eps_clip:       PPO clipping epsilon
- kl_coef:        KL penalty coefficient

Steps:
1. Group-normalize advantages: for each prompt, compute mean and std of rewards
   across the G responses, then advantage_g = (reward_g - mean) / (std + eps)
2. Broadcast per-response advantage to all tokens of that response
3. Compute PPO-clipped policy loss: let log_ratio = new_log_probs - old_log_probs,
   ratio = exp(log_ratio), then clip ratio and compute the standard PPO objective.
   (Note: the variable "ppo_kl" in some references is the negative log-ratio
   old_log_probs - new_log_probs, not an actual KL divergence.)
4. Add KL penalty term: kl_coef * mean_kl(policy || ref)
5. Return total loss (must be differentiable)

Reference: slime/utils/ppo_utils.py :: get_grpo_returns()
           slime/backends/megatron_utils/loss.py :: compute_advantages_and_returns() [grpo]
           slime/backends/megatron_utils/loss.py :: policy_loss_function()
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
    # TODO: Normalize rewards within each prompt group
    # mean and std computed per-prompt (dim=-1)
    raise NotImplementedError("Implement compute_group_advantages")


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

    The list of log_probs has length num_prompts * group_size, ordered as:
    [prompt_0_resp_0, prompt_0_resp_1, ..., prompt_0_resp_{G-1},
     prompt_1_resp_0, ...]

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
    # TODO: Implement GRPO loss
    # Step 1: Compute group-normalized advantages
    # Step 2: For each response, broadcast advantage to all tokens
    # Step 3: Compute PPO-clipped loss (ratio = exp(new_log_probs - old_log_probs))
    # Step 4: Compute KL penalty (policy vs reference, k3 estimator)
    # Step 5: Combine: loss = pg_loss + kl_coef * kl_loss
    raise NotImplementedError("Implement compute_grpo_loss")
