"""Tests for Exercise 04: GRPO Advantage Estimation"""

import importlib.util
import os
import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("ex04_solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_grpo_advantages = _mod.compute_grpo_advantages


class TestGRPOAdvantages:
    def test_basic_normalization(self):
        """Test that rewards are z-score normalized."""
        rewards = np.array([1.0, 3.0, 5.0])
        lengths = [2, 3, 4]
        advs = compute_grpo_advantages(rewards, lengths)

        # mean=3, std=sqrt(8/3) ~ 1.6330
        expected_norm = (rewards - 3.0) / np.std(rewards)
        for i, adv in enumerate(advs):
            assert adv.shape == (lengths[i],)
            np.testing.assert_allclose(adv, expected_norm[i], atol=1e-7)

    def test_identical_rewards_gives_zero(self):
        """When all rewards are identical, advantages should be zero."""
        rewards = np.array([5.0, 5.0, 5.0, 5.0])
        lengths = [3, 5, 2, 4]
        advs = compute_grpo_advantages(rewards, lengths)
        for adv in advs:
            np.testing.assert_allclose(adv, 0.0, atol=1e-10)

    def test_two_samples(self):
        """Simple two-sample group."""
        rewards = np.array([0.0, 2.0])
        lengths = [1, 1]
        advs = compute_grpo_advantages(rewards, lengths)
        # mean=1, std=1
        np.testing.assert_allclose(advs[0], [-1.0], atol=1e-7)
        np.testing.assert_allclose(advs[1], [1.0], atol=1e-7)

    def test_broadcast_to_tokens(self):
        """Each token in a response gets the same advantage value."""
        rewards = np.array([10.0, 0.0])
        lengths = [5, 10]
        advs = compute_grpo_advantages(rewards, lengths)
        assert advs[0].shape == (5,)
        assert advs[1].shape == (10,)
        # All tokens in response 0 should be the same value
        assert len(np.unique(advs[0])) == 1
        assert len(np.unique(advs[1])) == 1

    def test_mean_of_normalized_is_zero(self):
        """The raw normalized values should sum to approximately zero."""
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lengths = [1, 1, 1, 1, 1]
        advs = compute_grpo_advantages(rewards, lengths)
        all_vals = np.array([a[0] for a in advs])
        np.testing.assert_allclose(np.mean(all_vals), 0.0, atol=1e-7)

    def test_single_sample_group(self):
        """A group of one should have zero advantage (std=0)."""
        rewards = np.array([42.0])
        lengths = [7]
        advs = compute_grpo_advantages(rewards, lengths)
        np.testing.assert_allclose(advs[0], 0.0, atol=1e-10)
