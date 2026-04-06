"""Tests for Exercise 07: KL Divergence Penalty Loss"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_approx_kl = _mod.compute_approx_kl
compute_loss_with_kl_penalty = _mod.compute_loss_with_kl_penalty


def test_kl_zero_when_same():
    """KL should be ~0 when policy == reference for all types."""
    B, T = 3, 8
    lp = torch.randn(B, T)
    mask = torch.ones(B, T)

    for kl_type in ["k1", "k2", "k3"]:
        _, mean_kl = compute_approx_kl(lp, lp, mask, kl_type=kl_type)
        assert torch.isclose(mean_kl, torch.tensor(0.0), atol=1e-6), (
            f"KL ({kl_type}) should be 0 when same, got {mean_kl.item()}"
        )


def test_k3_non_negative():
    """k3 estimator should always produce non-negative KL."""
    B, T = 5, 20
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)
    mask = torch.ones(B, T)

    per_token_kl, mean_kl = compute_approx_kl(policy_lp, ref_lp, mask, kl_type="k3")
    assert (per_token_kl >= -1e-6).all(), "k3 KL should be non-negative per token"
    assert mean_kl.item() >= -1e-6, "k3 mean KL should be non-negative"


def test_k2_non_negative():
    """k2 estimator (squared) should always be non-negative."""
    B, T = 4, 10
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)
    mask = torch.ones(B, T)

    per_token_kl, mean_kl = compute_approx_kl(policy_lp, ref_lp, mask, kl_type="k2")
    assert (per_token_kl >= -1e-6).all()


def test_k1_known_value():
    """k1 = log_ratio = policy_lp - ref_lp."""
    policy_lp = torch.tensor([[0.5, -0.3]])
    ref_lp = torch.tensor([[0.2, -0.1]])
    mask = torch.ones(1, 2)

    per_token_kl, mean_kl = compute_approx_kl(policy_lp, ref_lp, mask, kl_type="k1")
    expected = torch.tensor([[0.3, -0.2]])
    assert torch.allclose(per_token_kl, expected, atol=1e-5)
    assert torch.isclose(mean_kl, torch.tensor(0.05), atol=1e-5)


def test_output_shapes():
    """per_token_kl should match input shape, mean_kl should be scalar."""
    B, T = 3, 6
    per_token_kl, mean_kl = compute_approx_kl(
        torch.randn(B, T), torch.randn(B, T), torch.ones(B, T)
    )
    assert per_token_kl.shape == (B, T)
    assert mean_kl.dim() == 0


def test_mask_respected():
    """Masked tokens should not affect mean KL."""
    policy_lp = torch.tensor([[1.0, 0.0, 99.0, 99.0]])
    ref_lp = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    _, mean_kl = compute_approx_kl(policy_lp, ref_lp, mask, kl_type="k1")
    # Only first two tokens: (1.0 + 0.0) / 2 = 0.5
    assert torch.isclose(mean_kl, torch.tensor(0.5), atol=1e-5)


def test_kl_penalty_increases_loss():
    """KL penalty should increase total loss when distributions differ."""
    B, T = 2, 5
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T) + 1.0  # Different
    mask = torch.ones(B, T)
    policy_loss = torch.tensor(1.0)

    total_loss, mean_kl = compute_loss_with_kl_penalty(
        policy_loss, policy_lp, ref_lp, mask, beta=0.1, kl_type="k3"
    )

    # k3 KL is non-negative, so total >= policy_loss
    assert total_loss.item() >= policy_loss.item() - 1e-6


def test_kl_penalty_zero_beta():
    """With beta=0, total_loss == policy_loss."""
    B, T = 2, 3
    policy_lp = torch.randn(B, T)
    ref_lp = torch.randn(B, T)
    mask = torch.ones(B, T)
    policy_loss = torch.tensor(2.5)

    total_loss, _ = compute_loss_with_kl_penalty(
        policy_loss, policy_lp, ref_lp, mask, beta=0.0
    )
    assert torch.isclose(total_loss, policy_loss, atol=1e-6)


def test_invalid_kl_type():
    """Unknown kl_type should raise ValueError."""
    try:
        compute_approx_kl(torch.randn(1, 1), torch.randn(1, 1), torch.ones(1, 1), kl_type="unknown")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
