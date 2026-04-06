"""Tests for Exercise 05: REINFORCE with Baseline"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("ex05_solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_discounted_returns = _mod.compute_discounted_returns
reinforce_with_baseline = _mod.reinforce_with_baseline


class TestDiscountedReturns:
    def test_single_reward_at_end(self):
        """Typical case: reward only at last token."""
        token_rewards = np.array([0.0, 0.0, 0.0, 1.0])
        gamma = 0.99
        returns = compute_discounted_returns(token_rewards, gamma)
        # G_3 = 1.0
        # G_2 = 0 + 0.99 * 1.0 = 0.99
        # G_1 = 0 + 0.99 * 0.99 = 0.9801
        # G_0 = 0 + 0.99 * 0.9801 = 0.970299
        np.testing.assert_allclose(returns[3], 1.0, atol=1e-7)
        np.testing.assert_allclose(returns[2], 0.99, atol=1e-7)
        np.testing.assert_allclose(returns[1], 0.9801, atol=1e-7)
        np.testing.assert_allclose(returns[0], 0.970299, atol=1e-7)

    def test_gamma_zero(self):
        """With gamma=0, G_t = r_t (no discounting of future)."""
        token_rewards = np.array([1.0, 2.0, 3.0])
        returns = compute_discounted_returns(token_rewards, gamma=0.0)
        np.testing.assert_allclose(returns, [1.0, 2.0, 3.0])

    def test_gamma_one(self):
        """With gamma=1, G_t = sum of all future rewards (no discount)."""
        token_rewards = np.array([1.0, 1.0, 1.0])
        returns = compute_discounted_returns(token_rewards, gamma=1.0)
        np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])

    def test_single_token(self):
        """Single token: return equals the reward."""
        token_rewards = np.array([5.0])
        returns = compute_discounted_returns(token_rewards, gamma=0.99)
        np.testing.assert_allclose(returns, [5.0])

    def test_all_zeros(self):
        """Zero rewards produce zero returns."""
        token_rewards = np.zeros(10)
        returns = compute_discounted_returns(token_rewards, gamma=0.99)
        np.testing.assert_allclose(returns, 0.0, atol=1e-10)


class TestReinforceWithBaseline:
    def test_identical_sequences_same_advantage(self):
        """If all sequences have identical rewards, they get the same advantages."""
        rewards_list = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        advs = reinforce_with_baseline(rewards_list, gamma=0.99)
        # Both sequences should have identical advantages
        np.testing.assert_allclose(advs[0], advs[1], atol=1e-10)
        # G_0 advantage should be 0 (baseline = G_0 for identical seqs)
        np.testing.assert_allclose(advs[0][0], 0.0, atol=1e-10)

    def test_baseline_subtraction(self):
        """Baseline should be the mean of G_0 across sequences."""
        # Seq 0: reward at end = 2.0, seq 1: reward at end = 0.0
        rewards_list = [
            np.array([0.0, 2.0]),
            np.array([0.0, 0.0]),
        ]
        gamma = 1.0
        advs = reinforce_with_baseline(rewards_list, gamma)
        # Returns for seq 0: [2.0, 2.0]
        # Returns for seq 1: [0.0, 0.0]
        # Baseline = mean(G_0) = mean(2.0, 0.0) = 1.0
        # Advantages for seq 0: [1.0, 1.0]
        # Advantages for seq 1: [-1.0, -1.0]
        np.testing.assert_allclose(advs[0], [1.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(advs[1], [-1.0, -1.0], atol=1e-7)

    def test_different_length_sequences(self):
        """Sequences can have different lengths."""
        rewards_list = [
            np.array([0.0, 0.0, 3.0]),
            np.array([0.0, 1.0]),
        ]
        gamma = 1.0
        advs = reinforce_with_baseline(rewards_list, gamma)
        # Returns for seq 0: [3, 3, 3]
        # Returns for seq 1: [1, 1]
        # Baseline = mean(3, 1) = 2
        np.testing.assert_allclose(advs[0], [1.0, 1.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(advs[1], [-1.0, -1.0], atol=1e-7)

    def test_mean_g0_is_zero_after_baseline(self):
        """The mean of G_0 advantages across the group should be zero."""
        rng = np.random.default_rng(42)
        rewards_list = [rng.standard_normal(5) for _ in range(10)]
        advs = reinforce_with_baseline(rewards_list, gamma=0.99)
        g0_advs = [a[0] for a in advs]
        np.testing.assert_allclose(np.mean(g0_advs), 0.0, atol=1e-10)

    def test_single_sequence(self):
        """Single sequence: baseline = G_0, so G_0 advantage = 0."""
        rewards_list = [np.array([0.0, 0.0, 5.0])]
        advs = reinforce_with_baseline(rewards_list, gamma=0.99)
        # Baseline = G_0 of the only sequence, so advantage at t=0 is 0
        np.testing.assert_allclose(advs[0][0], 0.0, atol=1e-10)
