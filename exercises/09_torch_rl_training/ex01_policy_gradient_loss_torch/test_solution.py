"""Tests for Exercise 01: PPO Clipped Policy Gradient Loss"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_ppo_clipped_loss = _mod.compute_ppo_clipped_loss


def test_basic_shape_and_type():
    """Loss should be scalar and clip_frac should be scalar."""
    B, T = 4, 10
    old_lp = torch.randn(B, T)
    new_lp = torch.randn(B, T, requires_grad=True)
    adv = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_ppo_clipped_loss(old_lp, new_lp, adv, mask)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert clip_frac.dim() == 0, "Clip fraction should be a scalar"


def test_differentiable():
    """loss.backward() should produce gradients on new_log_probs."""
    B, T = 2, 5
    old_lp = torch.randn(B, T)
    new_lp = torch.randn(B, T, requires_grad=True)
    adv = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, _ = compute_ppo_clipped_loss(old_lp, new_lp, adv, mask)
    loss.backward()
    assert new_lp.grad is not None, "Gradient should flow to new_log_probs"
    assert new_lp.grad.shape == new_lp.shape


def test_no_clip_when_ratio_is_one():
    """When old == new, ratio=1, no clipping should occur."""
    B, T = 3, 8
    lp = torch.randn(B, T)
    new_lp = lp.clone().detach().requires_grad_(True)
    adv = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_ppo_clipped_loss(lp, new_lp, adv, mask)
    assert torch.isclose(clip_frac, torch.tensor(0.0), atol=1e-6), (
        f"No clipping expected when ratio=1, got clip_frac={clip_frac.item()}"
    )


def test_clipping_happens_for_extreme_ratio():
    """When new_log_probs >> old_log_probs, ratio is large and clipping should occur."""
    B, T = 2, 10
    old_lp = torch.zeros(B, T)
    # Make ratio = exp(5) >> 1+eps, so clipping definitely happens
    new_lp = (5.0 * torch.ones(B, T)).requires_grad_(True)
    # Use positive advantages so surr1 is very negative (good) but surr2 clips ratio
    adv = torch.ones(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_ppo_clipped_loss(old_lp, new_lp, adv, mask)
    assert clip_frac.item() > 0.5, f"Expected high clip fraction, got {clip_frac.item()}"


def test_mask_respected():
    """Masked tokens should not contribute to loss."""
    B, T = 2, 6
    old_lp = torch.zeros(B, T)
    new_lp = torch.randn(B, T, requires_grad=True)
    adv = torch.randn(B, T)

    # Only first 3 tokens are valid
    mask = torch.zeros(B, T)
    mask[:, :3] = 1.0

    loss1, _ = compute_ppo_clipped_loss(old_lp, new_lp, adv, mask)

    # Compute manually with only first 3 tokens
    ratio = torch.exp(new_lp.detach()[:, :3] - old_lp[:, :3])
    s1 = -ratio * adv[:, :3]
    s2 = -torch.clamp(ratio, 0.8, 1.2) * adv[:, :3]
    expected = torch.maximum(s1, s2).mean()

    assert torch.isclose(loss1, expected, atol=1e-5), (
        f"Masked loss {loss1.item()} != expected {expected.item()}"
    )


def test_known_values():
    """Test with hand-computed values."""
    # ratio = exp(0.1) ~ 1.1052
    old_lp = torch.tensor([[0.0, 0.0]])
    new_lp = torch.tensor([[0.1, 0.1]], requires_grad=True)
    adv = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, clip_frac = compute_ppo_clipped_loss(old_lp, new_lp, adv, mask, eps_clip=0.2)

    ratio = torch.exp(torch.tensor(0.1))  # ~1.1052
    # Token 0: adv=1, surr1=-1.1052, surr2=-1.1052 (not clipped since 1.1052 < 1.2)
    # Token 1: adv=-1, surr1=1.1052, surr2=1.1052 (not clipped since 1.1052 < 1.2)
    expected_loss = (-ratio * 1.0 + ratio * 1.0) / 2.0  # = 0
    assert torch.isclose(loss, expected_loss, atol=1e-4), (
        f"Loss {loss.item()} != expected {expected_loss.item()}"
    )
