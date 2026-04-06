"""
Tests for Exercise 03: ALiBi Attention
"""

import importlib.util
import math
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_alibi_slopes = _mod.compute_alibi_slopes
compute_alibi_bias = _mod.compute_alibi_bias
alibi_attention = _mod.alibi_attention


class TestAlibiSlopes:
    def test_shape(self):
        slopes = compute_alibi_slopes(8)
        assert slopes.shape == (8,)

    def test_geometric_ratio(self):
        """Consecutive slopes should have a constant ratio."""
        slopes = compute_alibi_slopes(8)
        ratios = slopes[1:] / slopes[:-1]
        expected_ratio = 2.0 ** (-8.0 / 8.0)  # = 0.5 for 8 heads
        assert torch.allclose(ratios, torch.full_like(ratios, expected_ratio), atol=1e-6)

    def test_first_slope_value(self):
        slopes = compute_alibi_slopes(8)
        expected = 2.0 ** (-8.0 / 8.0)  # 2^(-1) = 0.5
        assert abs(slopes[0].item() - expected) < 1e-6

    def test_slopes_decrease(self):
        slopes = compute_alibi_slopes(16)
        # Each slope should be smaller than the previous
        assert (slopes[1:] < slopes[:-1]).all()

    def test_all_positive(self):
        slopes = compute_alibi_slopes(4)
        assert (slopes > 0).all()


class TestAlibiBias:
    def test_shape(self):
        bias = compute_alibi_bias(seq_len=10, num_heads=4)
        assert bias.shape == (4, 10, 10)

    def test_diagonal_is_zero(self):
        """At position (q, q), distance is 0, so bias should be 0."""
        bias = compute_alibi_bias(seq_len=8, num_heads=4)
        for h in range(4):
            diag = torch.diag(bias[h])
            assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_lower_triangle_non_positive(self):
        """For causal (k <= q), distance <= 0, slopes > 0, so bias <= 0."""
        bias = compute_alibi_bias(seq_len=8, num_heads=4)
        lower = torch.tril(bias, diagonal=0)
        assert (lower <= 1e-6).all()

    def test_bias_scales_with_distance(self):
        """Bias should be linear in distance for each head."""
        bias = compute_alibi_bias(seq_len=8, num_heads=2)
        slopes = compute_alibi_slopes(2)
        # Check bias[h, 4, 2] = slopes[h] * (2 - 4) = slopes[h] * (-2)
        for h in range(2):
            expected = slopes[h].item() * (2 - 4)
            assert abs(bias[h, 4, 2].item() - expected) < 1e-5


class TestAlibiAttention:
    def test_output_shape(self):
        B, H, S, D = 2, 4, 8, 16
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        bias = compute_alibi_bias(S, H)
        out = alibi_attention(Q, K, V, bias, causal_mask=True)
        assert out.shape == (B, H, S, D)

    def test_causal_first_position_attends_only_to_self(self):
        """With causal mask, position 0 can only attend to itself."""
        B, H, S, D = 1, 2, 4, 8
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        bias = compute_alibi_bias(S, H)
        out = alibi_attention(Q, K, V, bias, causal_mask=True)
        # Position 0 output should equal V at position 0 (only attends to self)
        assert torch.allclose(out[:, :, 0, :], V[:, :, 0, :], atol=1e-5)

    def test_no_causal_mask_attends_to_future(self):
        """Without causal mask, output at pos 0 should differ from V[0]."""
        B, H, S, D = 1, 2, 6, 8
        torch.manual_seed(42)
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)
        bias = compute_alibi_bias(S, H)
        out = alibi_attention(Q, K, V, bias, causal_mask=False)
        # Position 0 can attend to future, so output != V[0]
        assert not torch.allclose(out[:, :, 0, :], V[:, :, 0, :], atol=1e-3)

    def test_longer_sequence_extrapolation(self):
        """ALiBi should work on sequences longer than some reference length."""
        B, H, D = 1, 4, 16
        # "Train" length 8, "test" length 32
        for S in [8, 32]:
            Q = torch.randn(B, H, S, D)
            K = torch.randn(B, H, S, D)
            V = torch.randn(B, H, S, D)
            bias = compute_alibi_bias(S, H)
            out = alibi_attention(Q, K, V, bias, causal_mask=True)
            assert out.shape == (B, H, S, D)
            # No NaN or inf
            assert torch.isfinite(out).all()
