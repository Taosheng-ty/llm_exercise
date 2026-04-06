"""Tests for Exercise 05: Importance Sampling Ratios"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_token_is_ratio = _mod.compute_token_is_ratio
compute_sequence_is_ratio = _mod.compute_sequence_is_ratio
truncated_importance_sampling = _mod.truncated_importance_sampling


def test_token_ratio_shape():
    """Token ratio should match input shape."""
    B, T = 3, 8
    new_lp = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    ratio = compute_token_is_ratio(new_lp, old_lp)
    assert ratio.shape == (B, T)


def test_token_ratio_one_when_equal():
    """When new == old, ratio should be 1."""
    lp = torch.randn(2, 5)
    ratio = compute_token_is_ratio(lp, lp)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def test_token_ratio_positive():
    """IS ratios should always be positive."""
    ratio = compute_token_is_ratio(torch.randn(3, 10), torch.randn(3, 10))
    assert (ratio > 0).all()


def test_sequence_ratio_shape():
    """Sequence ratio should have shape (batch,)."""
    B, T = 4, 6
    new_lp = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    mask = torch.ones(B, T)
    seq_ratio = compute_sequence_is_ratio(new_lp, old_lp, mask)
    assert seq_ratio.shape == (B,)


def test_sequence_ratio_one_when_equal():
    """When new == old, sequence ratio should be 1."""
    lp = torch.randn(3, 5)
    mask = torch.ones(3, 5)
    seq_ratio = compute_sequence_is_ratio(lp, lp, mask)
    assert torch.allclose(seq_ratio, torch.ones(3), atol=1e-5)


def test_sequence_ratio_mask():
    """Masked tokens should not affect sequence ratio."""
    B, T = 1, 4
    new_lp = torch.tensor([[0.1, 0.2, 999.0, 999.0]])
    old_lp = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    seq_ratio = compute_sequence_is_ratio(new_lp, old_lp, mask)
    expected = torch.exp(torch.tensor(0.3))  # exp(0.1 + 0.2)
    assert torch.isclose(seq_ratio[0], expected, atol=1e-4)


def test_tis_shape():
    """TIS outputs should have correct shapes."""
    B, T = 3, 8
    new_lp = torch.randn(B, T)
    old_lp = torch.randn(B, T)
    mask = torch.ones(B, T)

    weights, clip_frac, raw = truncated_importance_sampling(new_lp, old_lp, mask)
    assert weights.shape == (B, T)
    assert clip_frac.dim() == 0
    assert raw.shape == (B, T)


def test_tis_no_clip_when_equal():
    """When new == old, ratio=1, no clipping needed."""
    lp = torch.randn(2, 5)
    mask = torch.ones(2, 5)

    weights, clip_frac, raw = truncated_importance_sampling(lp, lp, mask)
    assert torch.isclose(clip_frac, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-6)


def test_tis_clips_extreme_ratios():
    """Very different policies should produce clipped ratios."""
    B, T = 1, 4
    old_lp = torch.zeros(B, T)
    new_lp = torch.tensor([[10.0, -10.0, 0.0, 0.0]])  # Very different
    mask = torch.ones(B, T)

    weights, clip_frac, raw = truncated_importance_sampling(
        new_lp, old_lp, mask, clip_low=0.2, clip_high=5.0
    )

    # Token 0: exp(10) >> 5.0, should be clipped to 5.0
    assert torch.isclose(weights[0, 0], torch.tensor(5.0), atol=1e-4)
    # Token 1: exp(-10) << 0.2, should be clipped to 0.2
    assert torch.isclose(weights[0, 1], torch.tensor(0.2), atol=1e-4)
    # At least 2 of 4 tokens should be clipped
    assert clip_frac.item() >= 0.5


def test_tis_within_bounds():
    """All TIS weights should be within [clip_low, clip_high]."""
    B, T = 5, 20
    new_lp = torch.randn(B, T) * 3  # Large variance
    old_lp = torch.randn(B, T) * 3
    mask = torch.ones(B, T)

    weights, _, _ = truncated_importance_sampling(
        new_lp, old_lp, mask, clip_low=0.1, clip_high=10.0
    )
    assert (weights >= 0.1 - 1e-6).all()
    assert (weights <= 10.0 + 1e-6).all()
