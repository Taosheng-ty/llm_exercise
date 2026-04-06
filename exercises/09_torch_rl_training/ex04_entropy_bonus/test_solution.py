"""Tests for Exercise 04: Entropy Bonus"""

import importlib.util
import math
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_entropy_from_logits = _mod.compute_entropy_from_logits
compute_loss_with_entropy_bonus = _mod.compute_loss_with_entropy_bonus


def test_uniform_distribution_entropy():
    """Uniform distribution over V classes should have entropy = log(V)."""
    V = 100
    B, T = 1, 1
    # Uniform logits -> uniform distribution
    logits = torch.zeros(B, T, V)
    mask = torch.ones(B, T)

    entropy = compute_entropy_from_logits(logits, mask)
    expected = math.log(V)
    assert torch.isclose(entropy, torch.tensor(expected), atol=1e-4), (
        f"Entropy {entropy.item()} != expected {expected}"
    )


def test_deterministic_distribution_entropy():
    """One-hot distribution should have entropy ~ 0."""
    B, T, V = 1, 1, 10
    logits = torch.full((B, T, V), -1e6)
    logits[:, :, 0] = 0.0  # All probability on class 0
    mask = torch.ones(B, T)

    entropy = compute_entropy_from_logits(logits, mask)
    assert entropy.item() < 0.01, f"Entropy should be ~0 for deterministic, got {entropy.item()}"


def test_entropy_non_negative():
    """Entropy should always be non-negative."""
    B, T, V = 4, 10, 50
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T)

    entropy = compute_entropy_from_logits(logits, mask)
    assert entropy.item() >= -1e-6, f"Entropy should be non-negative, got {entropy.item()}"


def test_mask_respected():
    """Masked tokens should not affect entropy."""
    B, T, V = 1, 4, 10
    logits = torch.randn(B, T, V)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    entropy_masked = compute_entropy_from_logits(logits, mask)

    # Manual: compute entropy only for first 2 tokens
    log_probs = torch.nn.functional.log_softmax(logits[:, :2], dim=-1)
    probs = log_probs.exp()
    per_token = -(probs * log_probs).sum(dim=-1)
    expected = per_token.mean()

    assert torch.isclose(entropy_masked, expected, atol=1e-5)


def test_entropy_bonus_reduces_loss():
    """Adding entropy bonus should reduce total loss (since entropy > 0)."""
    B, T, V = 2, 5, 20
    logits = torch.randn(B, T, V, requires_grad=True)
    mask = torch.ones(B, T)
    policy_loss = torch.tensor(1.0, requires_grad=True)

    total_loss, mean_entropy = compute_loss_with_entropy_bonus(
        policy_loss, logits, mask, entropy_coef=0.01
    )

    # Since entropy > 0 and coef > 0, total_loss < policy_loss
    assert total_loss.item() < policy_loss.item()
    assert mean_entropy.item() > 0


def test_entropy_bonus_differentiable():
    """Total loss should be differentiable w.r.t. logits."""
    B, T, V = 2, 3, 10
    logits = torch.randn(B, T, V, requires_grad=True)
    mask = torch.ones(B, T)
    policy_loss = torch.tensor(0.5)

    total_loss, _ = compute_loss_with_entropy_bonus(
        policy_loss, logits, mask, entropy_coef=0.01
    )
    total_loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_zero_coef_no_change():
    """With entropy_coef=0, total_loss == policy_loss."""
    B, T, V = 1, 2, 5
    logits = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    policy_loss = torch.tensor(3.14)

    total_loss, _ = compute_loss_with_entropy_bonus(
        policy_loss, logits, mask, entropy_coef=0.0
    )
    assert torch.isclose(total_loss, policy_loss, atol=1e-6)
