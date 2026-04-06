"""Tests for Exercise 04: Dual-Clip PPO Loss."""
import numpy as np
import pytest

from .solution import dual_clip_ppo_loss


class TestDualClipPPOLoss:
    def test_no_policy_change(self):
        """When log_probs == old_log_probs, ratio=1, no clipping occurs."""
        lp = np.array([0.0, 0.0, 0.0])
        advantages = np.array([1.0, -1.0, 0.5])
        loss, clip_frac = dual_clip_ppo_loss(lp, lp, advantages)
        # ratio=1, L1 = -1*adv, L2 = -clip(1, 0.8, 1.2)*adv = -adv
        # No clipping: L1 == L2
        expected = -advantages
        # For negative advantage, dual clip: min(-c*adv, L_clip)
        # adv=-1: min(-3*(-1), 1) = min(3, 1) = 1
        expected[1] = min(-3.0 * advantages[1], -advantages[1])
        np.testing.assert_allclose(loss, expected, atol=1e-6)
        assert clip_frac == 0.0

    def test_standard_clipping_positive_advantage(self):
        """With positive advantage and large ratio, standard clip should activate."""
        # ratio = exp(0.5) ~ 1.649, clip to 1.2
        log_probs = np.array([0.5])
        old_log_probs = np.array([0.0])
        advantages = np.array([1.0])  # positive advantage
        loss, clip_frac = dual_clip_ppo_loss(
            log_probs, old_log_probs, advantages, eps_clip=0.2
        )
        ratio = np.exp(0.5)
        l1 = -ratio * 1.0
        l2 = -1.2 * 1.0
        expected = max(l1, l2)  # l2 > l1 since l2 is less negative
        np.testing.assert_allclose(loss[0], expected, atol=1e-6)
        assert clip_frac == 1.0

    def test_dual_clip_negative_advantage(self):
        """With negative advantage and large ratio, dual clip should limit loss."""
        # ratio = exp(1.0) ~ 2.718, clip to 1.2
        log_probs = np.array([1.0])
        old_log_probs = np.array([0.0])
        advantages = np.array([-1.0])
        eps_clip_c = 3.0
        loss, clip_frac = dual_clip_ppo_loss(
            log_probs, old_log_probs, advantages, eps_clip=0.2, eps_clip_c=eps_clip_c
        )
        ratio = np.exp(1.0)
        l1 = -ratio * (-1.0)  # = ratio = 2.718
        l2 = -1.2 * (-1.0)   # = 1.2
        l_clip = max(l1, l2)  # = 2.718
        l3 = -eps_clip_c * (-1.0)  # = 3.0
        expected = min(l3, l_clip)  # = min(3.0, 2.718) = 2.718
        np.testing.assert_allclose(loss[0], expected, atol=1e-6)

    def test_dual_clip_caps_loss(self):
        """When standard PPO loss exceeds c*|advantage|, dual clip should cap it."""
        # We need L_clip > L3 = -c * adv = c * |adv|
        # With very large ratio and negative advantage
        log_probs = np.array([3.0])  # ratio = exp(3) ~ 20.09
        old_log_probs = np.array([0.0])
        advantages = np.array([-1.0])
        eps_clip_c = 3.0
        loss, _ = dual_clip_ppo_loss(
            log_probs, old_log_probs, advantages, eps_clip=0.2, eps_clip_c=eps_clip_c
        )
        ratio = np.exp(3.0)
        l1 = -ratio * (-1.0)   # = 20.09
        l2 = -1.2 * (-1.0)     # = 1.2
        l_clip = max(l1, l2)    # = 20.09
        l3 = -eps_clip_c * (-1.0)  # = 3.0
        expected = min(l3, l_clip)  # = 3.0 (dual clip caps it)
        np.testing.assert_allclose(loss[0], expected, atol=1e-6)

    def test_output_shape(self):
        """Output loss shape should match input shape."""
        n = 10
        lp = np.random.randn(n) * 0.1
        old_lp = np.random.randn(n) * 0.1
        adv = np.random.randn(n)
        loss, clip_frac = dual_clip_ppo_loss(lp, old_lp, adv)
        assert loss.shape == (n,)
        assert 0.0 <= clip_frac <= 1.0

    def test_eps_clip_c_must_be_greater_than_one(self):
        """Should raise assertion error if eps_clip_c <= 1."""
        with pytest.raises(AssertionError):
            dual_clip_ppo_loss(
                np.array([0.0]), np.array([0.0]), np.array([1.0]),
                eps_clip_c=0.5,
            )

    def test_clip_fraction_reflects_clipping(self):
        """Clip fraction should be between 0 and 1 and reflect actual clipping."""
        # All tokens have ratio=1 (no change) -> no clipping
        n = 5
        lp = np.zeros(n)
        adv = np.ones(n)
        _, clip_frac = dual_clip_ppo_loss(lp, lp, adv)
        assert clip_frac == 0.0

    def test_positive_advantage_unaffected_by_dual_clip(self):
        """Dual clip should not affect tokens with positive advantage."""
        log_probs = np.array([0.5])
        old_log_probs = np.array([0.0])
        advantages = np.array([2.0])

        loss_dual, _ = dual_clip_ppo_loss(
            log_probs, old_log_probs, advantages, eps_clip=0.2, eps_clip_c=3.0
        )
        # Standard PPO would give the same result for positive advantages
        ratio = np.exp(0.5)
        l1 = -ratio * 2.0
        l2 = -np.clip(ratio, 0.8, 1.2) * 2.0
        expected = max(l1, l2)
        np.testing.assert_allclose(loss_dual[0], expected, atol=1e-6)
