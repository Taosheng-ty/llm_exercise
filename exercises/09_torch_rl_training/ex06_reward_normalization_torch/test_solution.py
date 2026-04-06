"""Tests for Exercise 06: Reward Normalization"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RewardNormalizer = _mod.RewardNormalizer


def test_basic_normalization():
    """Normalized output should have approximately zero mean."""
    normalizer = RewardNormalizer(momentum=1.0)  # Full update
    rewards = torch.randn(100) * 5 + 10  # mean~10, std~5

    normalized = normalizer.update_and_normalize(rewards)
    assert abs(normalized.mean().item()) < 0.5, (
        f"Normalized mean should be ~0, got {normalized.mean().item()}"
    )


def test_output_shape():
    """Output shape should match input shape."""
    normalizer = RewardNormalizer()

    r1d = torch.randn(10)
    out1d = normalizer.update_and_normalize(r1d)
    assert out1d.shape == r1d.shape

    r2d = torch.randn(4, 8)
    out2d = normalizer.update_and_normalize(r2d)
    assert out2d.shape == r2d.shape


def test_first_batch_initializes():
    """First batch should initialize running stats."""
    normalizer = RewardNormalizer(momentum=0.1)
    assert not normalizer.initialized

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalizer.update_and_normalize(rewards)

    assert normalizer.initialized
    assert abs(normalizer.running_mean - 3.0) < 0.01


def test_ema_update():
    """Running mean should move toward new batch mean."""
    normalizer = RewardNormalizer(momentum=0.5)

    # First batch: mean = 0
    normalizer.update_and_normalize(torch.zeros(10))
    assert abs(normalizer.running_mean) < 0.01

    # Second batch: mean = 10
    normalizer.update_and_normalize(torch.ones(10) * 10)
    # EMA: 0.5 * 0 + 0.5 * 10 = 5
    assert abs(normalizer.running_mean - 5.0) < 0.5


def test_zero_variance_handling():
    """Single-element batch (zero variance) should not crash."""
    normalizer = RewardNormalizer()
    rewards = torch.tensor([5.0])
    normalized = normalizer.update_and_normalize(rewards)
    assert not torch.isnan(normalized).any()
    assert not torch.isinf(normalized).any()


def test_constant_rewards():
    """Constant rewards should produce zero-mean output."""
    normalizer = RewardNormalizer(momentum=1.0)
    rewards = torch.ones(20) * 42.0
    normalized = normalizer.update_and_normalize(rewards)
    # All same value, normalized to 0
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-3)


def test_normalize_without_update():
    """normalize() should use current stats without updating them."""
    normalizer = RewardNormalizer(momentum=1.0)
    normalizer.update_and_normalize(torch.randn(50))

    old_mean = normalizer.running_mean
    old_var = normalizer.running_var

    # Call normalize (no update)
    normalizer.normalize(torch.randn(50) * 100)

    assert normalizer.running_mean == old_mean
    assert normalizer.running_var == old_var


def test_multiple_batches_stability():
    """After many batches, normalizer should produce reasonable outputs."""
    normalizer = RewardNormalizer(momentum=0.1)

    for _ in range(100):
        rewards = torch.randn(32) * 3 + 7  # mean=7, std=3
        normalized = normalizer.update_and_normalize(rewards)

    # Running mean should be close to 7
    assert abs(normalizer.running_mean - 7.0) < 1.0
    # Normalized output should have approximately unit variance
    final_normalized = normalizer.update_and_normalize(torch.randn(1000) * 3 + 7)
    assert 0.1 < final_normalized.std().item() < 5.0
