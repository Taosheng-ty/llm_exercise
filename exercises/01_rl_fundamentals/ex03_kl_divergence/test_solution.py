"""Tests for Exercise 03: KL Divergence Approximation Methods"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("ex03_solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_approx_kl = _mod.compute_approx_kl


class TestKLDivergence:
    def test_equal_distributions_all_zero(self):
        """When distributions are identical, all KL approximations should be 0."""
        log_probs = np.array([-1.0, -2.0, -0.5])
        for kl_type in ["k1", "k2", "k3"]:
            kl = compute_approx_kl(log_probs, log_probs, kl_type)
            np.testing.assert_allclose(kl, 0.0, atol=1e-10, err_msg=f"Failed for {kl_type}")

    def test_k1_can_be_negative(self):
        """k1 = log_ratio, which can be negative when new < old."""
        log_probs_new = np.array([-2.0])
        log_probs_old = np.array([-1.0])
        kl = compute_approx_kl(log_probs_new, log_probs_old, "k1")
        assert kl[0] < 0, "k1 should be negative when new prob < old prob"
        np.testing.assert_allclose(kl, [-1.0])

    def test_k2_always_non_negative(self):
        """k2 = 0.5 * log_ratio^2, which is always >= 0."""
        rng = np.random.default_rng(42)
        log_probs_new = rng.standard_normal(100)
        log_probs_old = rng.standard_normal(100)
        kl = compute_approx_kl(log_probs_new, log_probs_old, "k2")
        assert np.all(kl >= 0), "k2 should always be non-negative"

    def test_k3_always_non_negative(self):
        """k3 = exp(-log_ratio) - 1 + log_ratio, which is always >= 0."""
        rng = np.random.default_rng(123)
        log_probs_new = rng.standard_normal(100)
        log_probs_old = rng.standard_normal(100)
        kl = compute_approx_kl(log_probs_new, log_probs_old, "k3")
        assert np.all(kl >= -1e-10), "k3 should always be non-negative"

    def test_k2_hand_computed(self):
        """Verify k2 with hand-computed values."""
        log_probs_new = np.array([0.0, -1.0])
        log_probs_old = np.array([-1.0, 0.0])
        kl = compute_approx_kl(log_probs_new, log_probs_old, "k2")
        # log_ratio = [1.0, -1.0], k2 = [0.5, 0.5]
        np.testing.assert_allclose(kl, [0.5, 0.5])

    def test_k3_hand_computed(self):
        """Verify k3 with hand-computed values."""
        log_probs_new = np.array([0.0])
        log_probs_old = np.array([-1.0])
        kl = compute_approx_kl(log_probs_new, log_probs_old, "k3")
        # log_ratio = 1.0, neg_log_ratio = -1.0
        # k3 = exp(-1) - 1 - (-1) = exp(-1) = 0.36787944...
        np.testing.assert_allclose(kl, [np.exp(-1.0)], atol=1e-7)

    def test_unknown_type_raises(self):
        """Unknown kl_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            compute_approx_kl(np.array([0.0]), np.array([0.0]), "k99")

    def test_k3_both_directions_non_negative(self):
        """k3 should be non-negative regardless of direction."""
        log_probs_a = np.array([-0.5, -1.5, -2.0])
        log_probs_b = np.array([-1.0, -0.5, -3.0])
        kl_ab = compute_approx_kl(log_probs_a, log_probs_b, "k3")
        kl_ba = compute_approx_kl(log_probs_b, log_probs_a, "k3")
        assert np.all(kl_ab >= -1e-10)
        assert np.all(kl_ba >= -1e-10)
