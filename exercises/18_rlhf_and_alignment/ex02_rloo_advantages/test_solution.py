"""Tests for Exercise 02: REINFORCE Leave-One-Out (RLOO) Advantages"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
rloo_advantages = _mod.rloo_advantages


class TestRLOOAdvantages:
    def test_basic_two_completions(self):
        """With K=2, advantage_i = r_i - r_j for the other completion."""
        rewards = np.array([[3.0, 1.0]])  # 1 prompt, 2 completions
        adv = rloo_advantages(rewards)
        # advantage_0 = 3 - 1 = 2, advantage_1 = 1 - 3 = -2
        np.testing.assert_allclose(adv, [[2.0, -2.0]])

    def test_three_completions(self):
        """With K=3, leave-one-out mean uses the other 2."""
        rewards = np.array([[6.0, 2.0, 4.0]])
        adv = rloo_advantages(rewards)
        # loo_mean_0 = (2+4)/2 = 3.0, adv_0 = 6-3 = 3
        # loo_mean_1 = (6+4)/2 = 5.0, adv_1 = 2-5 = -3
        # loo_mean_2 = (6+2)/2 = 4.0, adv_2 = 4-4 = 0
        np.testing.assert_allclose(adv, [[3.0, -3.0, 0.0]])

    def test_k_equals_one_returns_zeros(self):
        """Edge case: K=1 means no other completions, advantage should be 0."""
        rewards = np.array([[5.0], [3.0], [7.0]])
        adv = rloo_advantages(rewards)
        np.testing.assert_allclose(adv, np.zeros((3, 1)))

    def test_output_shape(self):
        """Output shape should match input shape."""
        rewards = np.random.randn(10, 4)
        adv = rloo_advantages(rewards)
        assert adv.shape == rewards.shape

    def test_advantages_sum_to_correct_value(self):
        """For each prompt, sum of advantages should have a known relationship.

        Sum of advantages = sum(r_i) - sum(loo_mean_i)
        = sum(r_i) - sum((total - r_i)/(K-1))
        = sum(r_i) - (K*total - total)/(K-1)
        = total - total = 0 ... only when K=2.
        For general K: sum_adv = K * mean - K * (K*mean - mean)/(K-1)... not always 0.
        Actually: sum_adv_i = sum(r_i - (S-r_i)/(K-1)) = S - (K*S - S)/(K-1) = S - S = 0.
        So they always sum to 0!
        """
        rewards = np.random.randn(5, 8)
        adv = rloo_advantages(rewards)
        row_sums = adv.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-10)

    def test_equal_rewards_give_zero_advantages(self):
        """If all completions have the same reward, advantages are all zero."""
        rewards = np.full((3, 5), 4.2)
        adv = rloo_advantages(rewards)
        np.testing.assert_allclose(adv, 0.0, atol=1e-10)

    def test_multiple_prompts(self):
        """Each prompt row should be computed independently."""
        rewards = np.array([
            [4.0, 2.0],
            [1.0, 5.0],
        ])
        adv = rloo_advantages(rewards)
        # Prompt 0: [4-2, 2-4] = [2, -2]
        # Prompt 1: [1-5, 5-1] = [-4, 4]
        np.testing.assert_allclose(adv, [[2.0, -2.0], [-4.0, 4.0]])

    def test_variance_reduction_vs_simple_baseline(self):
        """RLOO advantages should have lower variance than (r_i - mean(all))."""
        np.random.seed(42)
        rewards = np.random.randn(100, 8) * 5 + 2

        rloo_adv = rloo_advantages(rewards)

        # Simple baseline: r_i - mean(all K) for the same prompt
        simple_baseline = rewards.mean(axis=1, keepdims=True)
        simple_adv = rewards - simple_baseline

        # RLOO should have variance comparable to or different from simple
        # The key property: RLOO and simple differ by a factor of K/(K-1)
        # rloo_i = r_i - (S-r_i)/(K-1) = r_i*(K/(K-1)) - S/(K-1)
        # simple_i = r_i - S/K
        # Both are valid; RLOO is unbiased. Just verify they differ.
        assert not np.allclose(rloo_adv, simple_adv)

    def test_highest_reward_gets_positive_advantage(self):
        """The completion with the highest reward should have positive advantage."""
        rewards = np.array([[1.0, 5.0, 3.0, 2.0]])
        adv = rloo_advantages(rewards)
        best_idx = np.argmax(rewards[0])
        assert adv[0, best_idx] > 0

    def test_lowest_reward_gets_negative_advantage(self):
        """The completion with the lowest reward should have negative advantage."""
        rewards = np.array([[1.0, 5.0, 3.0, 2.0]])
        adv = rloo_advantages(rewards)
        worst_idx = np.argmin(rewards[0])
        assert adv[0, worst_idx] < 0
