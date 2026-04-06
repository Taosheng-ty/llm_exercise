"""Tests for Exercise 08: GRPO Loss Function"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_group_advantages = _mod.compute_group_advantages
compute_grpo_loss = _mod.compute_grpo_loss


# --- Tests for compute_group_advantages ---


def test_group_advantages_shape():
    """Advantages shape should match rewards shape."""
    rewards = torch.randn(4, 8)
    adv = compute_group_advantages(rewards)
    assert adv.shape == rewards.shape


def test_group_advantages_zero_mean():
    """Advantages should have approximately zero mean per prompt."""
    rewards = torch.randn(5, 10) * 3 + 7
    adv = compute_group_advantages(rewards)
    per_prompt_mean = adv.mean(dim=-1)
    assert torch.allclose(per_prompt_mean, torch.zeros_like(per_prompt_mean), atol=1e-5)


def test_group_advantages_unit_std():
    """Advantages should have approximately unit std per prompt."""
    rewards = torch.randn(5, 10) * 3 + 7
    adv = compute_group_advantages(rewards)
    per_prompt_std = adv.std(dim=-1)
    assert torch.allclose(per_prompt_std, torch.ones_like(per_prompt_std), atol=0.2)


def test_group_advantages_ordering_preserved():
    """Higher reward should produce higher advantage."""
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    adv = compute_group_advantages(rewards)
    assert adv[0, 0] < adv[0, 1] < adv[0, 2]


# --- Tests for compute_grpo_loss ---


def _make_grpo_inputs(num_prompts=2, group_size=3, seq_len=5, requires_grad=True):
    """Helper to create GRPO inputs."""
    total = num_prompts * group_size
    new_lp = [torch.randn(seq_len, requires_grad=requires_grad) for _ in range(total)]
    old_lp = [torch.randn(seq_len) for _ in range(total)]
    ref_lp = [torch.randn(seq_len) for _ in range(total)]
    rewards = torch.randn(num_prompts, group_size)
    masks = [torch.ones(seq_len) for _ in range(total)]
    return new_lp, old_lp, ref_lp, rewards, masks


def test_grpo_loss_scalar():
    """GRPO loss should be a scalar."""
    new_lp, old_lp, ref_lp, rewards, masks = _make_grpo_inputs()
    loss, metrics = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=3)
    assert loss.dim() == 0


def test_grpo_loss_differentiable():
    """GRPO loss should be differentiable w.r.t. new_log_probs."""
    new_lp, old_lp, ref_lp, rewards, masks = _make_grpo_inputs()
    loss, _ = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=3)
    loss.backward()
    for nlp in new_lp:
        assert nlp.grad is not None, "Gradient should flow to new_log_probs"


def test_grpo_loss_metrics_present():
    """Metrics dict should contain expected keys."""
    new_lp, old_lp, ref_lp, rewards, masks = _make_grpo_inputs()
    _, metrics = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=3)
    for key in ["pg_loss", "kl_loss", "mean_advantage", "clip_frac"]:
        assert key in metrics, f"Missing metric: {key}"


def test_grpo_zero_kl_coef():
    """With kl_coef=0, total loss should equal pg_loss."""
    new_lp, old_lp, ref_lp, rewards, masks = _make_grpo_inputs(requires_grad=False)
    loss, metrics = compute_grpo_loss(
        new_lp, old_lp, ref_lp, rewards, masks, group_size=3, kl_coef=0.0
    )
    assert torch.isclose(loss, metrics["pg_loss"], atol=1e-5)


def test_grpo_same_policy_low_kl():
    """When new == old == ref, KL should be ~0."""
    num_prompts, group_size, seq_len = 2, 3, 5
    total = num_prompts * group_size
    lp = [torch.randn(seq_len) for _ in range(total)]
    # All three policies are the same
    new_lp = [x.clone().requires_grad_(True) for x in lp]
    old_lp = [x.clone() for x in lp]
    ref_lp = [x.clone() for x in lp]
    rewards = torch.randn(num_prompts, group_size)
    masks = [torch.ones(seq_len) for _ in range(total)]

    _, metrics = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=group_size)
    assert metrics["kl_loss"].item() < 0.01, f"KL should be ~0, got {metrics['kl_loss'].item()}"
    assert metrics["clip_frac"].item() < 0.01, f"Clip frac should be ~0, got {metrics['clip_frac'].item()}"


def test_grpo_variable_seq_len():
    """GRPO should handle variable sequence lengths."""
    num_prompts, group_size = 2, 2
    total = num_prompts * group_size
    seq_lens = [3, 5, 4, 7]

    new_lp = [torch.randn(sl, requires_grad=True) for sl in seq_lens]
    old_lp = [torch.randn(sl) for sl in seq_lens]
    ref_lp = [torch.randn(sl) for sl in seq_lens]
    rewards = torch.randn(num_prompts, group_size)
    masks = [torch.ones(sl) for sl in seq_lens]

    loss, metrics = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=group_size)
    assert loss.dim() == 0
    loss.backward()
    for nlp in new_lp:
        assert nlp.grad is not None


def test_grpo_equal_rewards_zero_advantage():
    """When all rewards in a group are equal, advantages should be 0, giving low pg_loss."""
    num_prompts, group_size, seq_len = 2, 4, 5
    total = num_prompts * group_size

    # Same log_probs for old and new
    lp = [torch.randn(seq_len) for _ in range(total)]
    new_lp = [x.clone().requires_grad_(True) for x in lp]
    old_lp = [x.clone() for x in lp]
    ref_lp = [x.clone() for x in lp]

    # All rewards equal within each group
    rewards = torch.tensor([[5.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]])
    masks = [torch.ones(seq_len) for _ in range(total)]

    loss, metrics = compute_grpo_loss(new_lp, old_lp, ref_lp, rewards, masks, group_size=group_size)
    # With equal rewards, advantages are 0, so pg_loss ~ 0
    assert abs(metrics["pg_loss"].item()) < 0.01, (
        f"With equal rewards, pg_loss should be ~0, got {metrics['pg_loss'].item()}"
    )
