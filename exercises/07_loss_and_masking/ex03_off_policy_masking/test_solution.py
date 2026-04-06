"""Tests for Exercise 03: Off-Policy Sequence Masking (OPSM)."""
import numpy as np
import pytest

from .solution import compute_opsm_mask


class TestComputeOpsmMask:
    def test_no_masking_positive_advantages(self):
        """Sequences with positive advantages should never be masked."""
        seq_kl = np.array([0.5, 1.0, 2.0])
        advantages = np.array([1.0, 0.5, 0.1])
        delta = 0.1
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        np.testing.assert_array_equal(mask, [1.0, 1.0, 1.0])
        assert clip_frac == 0.0

    def test_no_masking_low_kl(self):
        """Even with negative advantages, low KL should not be masked."""
        seq_kl = np.array([0.01, 0.05, 0.02])
        advantages = np.array([-1.0, -2.0, -0.5])
        delta = 0.1
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        np.testing.assert_array_equal(mask, [1.0, 1.0, 1.0])
        assert clip_frac == 0.0

    def test_masking_negative_adv_high_kl(self):
        """Sequences with negative advantage AND high KL should be masked."""
        seq_kl = np.array([0.5, 0.01, 0.8])
        advantages = np.array([-1.0, -2.0, -0.5])
        delta = 0.1
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        # Seq 0: adv<0 and kl>0.1 -> masked (0)
        # Seq 1: adv<0 but kl<=0.1 -> kept (1)
        # Seq 2: adv<0 and kl>0.1 -> masked (0)
        np.testing.assert_array_equal(mask, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(clip_frac, 2.0 / 3.0, atol=1e-6)

    def test_mixed_conditions(self):
        """Test a mix of all conditions."""
        seq_kl = np.array([0.5, 0.01, 0.3, 0.8])
        advantages = np.array([1.0, -2.0, -0.5, 0.3])
        delta = 0.2
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        # Seq 0: adv>0 -> keep (1)
        # Seq 1: adv<0, kl<delta -> keep (1)
        # Seq 2: adv<0, kl>delta -> mask (0)
        # Seq 3: adv>0 -> keep (1)
        np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(clip_frac, 0.25, atol=1e-6)

    def test_boundary_kl_equals_delta(self):
        """When KL exactly equals delta, should NOT be masked (> not >=)."""
        seq_kl = np.array([0.1])
        advantages = np.array([-1.0])
        delta = 0.1
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        np.testing.assert_array_equal(mask, [1.0])
        assert clip_frac == 0.0

    def test_boundary_advantage_zero(self):
        """When advantage is exactly 0, should NOT be masked (< not <=)."""
        seq_kl = np.array([0.5])
        advantages = np.array([0.0])
        delta = 0.1
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta)
        np.testing.assert_array_equal(mask, [1.0])
        assert clip_frac == 0.0

    def test_output_shape(self):
        """Mask shape should match input shape."""
        n = 10
        seq_kl = np.random.rand(n)
        advantages = np.random.randn(n)
        mask, clip_frac = compute_opsm_mask(seq_kl, advantages, delta=0.5)
        assert mask.shape == (n,)
        assert 0.0 <= clip_frac <= 1.0
