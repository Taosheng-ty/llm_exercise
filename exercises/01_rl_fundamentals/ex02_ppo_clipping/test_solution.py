"""Tests for Exercise 02: PPO Clipped Surrogate Objective"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("ex02_solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_policy_loss = _mod.compute_policy_loss


class TestPPOClipping:
    def test_no_change_in_policy(self):
        """When log_probs are identical, ratio=1, no clipping needed."""
        log_probs = np.array([0.0, -1.0, -2.0])
        advantages = np.array([1.0, -1.0, 0.5])
        loss, clip_frac = compute_policy_loss(log_probs, log_probs, advantages, eps_clip=0.2)
        # ratio=1, loss = -1 * advantages
        np.testing.assert_allclose(loss, -advantages, atol=1e-7)
        assert clip_frac == 0.0

    def test_positive_advantage_ratio_too_high(self):
        """Positive advantage with ratio > 1+eps should be clipped."""
        # ratio = exp(0.5) ~ 1.6487, eps=0.2 -> clip to 1.2
        log_probs_new = np.array([0.5])
        log_probs_old = np.array([0.0])
        advantages = np.array([1.0])
        loss, clip_frac = compute_policy_loss(log_probs_new, log_probs_old, advantages, eps_clip=0.2)
        # unclipped: -1.6487 * 1 = -1.6487
        # clipped:   -1.2 * 1 = -1.2
        # max(-1.6487, -1.2) = -1.2
        np.testing.assert_allclose(loss, [-1.2], atol=1e-4)
        assert clip_frac == 1.0

    def test_negative_advantage_ratio_too_low(self):
        """Negative advantage with ratio < 1-eps should be clipped."""
        # ratio = exp(-0.5) ~ 0.6065, eps=0.2 -> clip to 0.8
        log_probs_new = np.array([-0.5])
        log_probs_old = np.array([0.0])
        advantages = np.array([-1.0])
        loss, clip_frac = compute_policy_loss(log_probs_new, log_probs_old, advantages, eps_clip=0.2)
        # unclipped: -0.6065 * (-1) = 0.6065
        # clipped:   -0.8 * (-1) = 0.8
        # max(0.6065, 0.8) = 0.8
        np.testing.assert_allclose(loss, [0.8], atol=1e-4)
        assert clip_frac == 1.0

    def test_ratio_within_clip_range(self):
        """When ratio is within [1-eps, 1+eps], no clipping occurs."""
        # ratio = exp(0.1) ~ 1.105, which is within [0.8, 1.2]
        log_probs_new = np.array([0.1])
        log_probs_old = np.array([0.0])
        advantages = np.array([2.0])
        loss, clip_frac = compute_policy_loss(log_probs_new, log_probs_old, advantages, eps_clip=0.2)
        ratio = np.exp(0.1)
        np.testing.assert_allclose(loss, [-ratio * 2.0], atol=1e-4)
        assert clip_frac == 0.0

    def test_mixed_clipping(self):
        """Test a batch with some clipped and some unclipped samples."""
        log_probs_new = np.array([0.0, 1.0, -1.0])
        log_probs_old = np.array([0.0, 0.0, 0.0])
        advantages = np.array([1.0, 1.0, -1.0])
        eps_clip = 0.2
        loss, clip_frac = compute_policy_loss(log_probs_new, log_probs_old, advantages, eps_clip)
        # Sample 0: ratio=1.0, within range, no clip
        # Sample 1: ratio=e^1~2.718, positive adv -> clip to 1.2
        # Sample 2: ratio=e^-1~0.368, negative adv -> clip to 0.8
        assert clip_frac == pytest.approx(2.0 / 3.0, abs=1e-7)

    def test_zero_advantage(self):
        """With zero advantage, loss should be zero regardless of ratio."""
        log_probs_new = np.array([2.0, -2.0])
        log_probs_old = np.array([0.0, 0.0])
        advantages = np.array([0.0, 0.0])
        loss, _ = compute_policy_loss(log_probs_new, log_probs_old, advantages, eps_clip=0.2)
        np.testing.assert_allclose(loss, [0.0, 0.0], atol=1e-7)
