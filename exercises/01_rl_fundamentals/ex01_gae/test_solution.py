"""Tests for Exercise 01: Generalized Advantage Estimation (GAE)"""

import importlib.util
import os
import numpy as np
import pytest

# Load solution from the same directory via path-based import
_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("ex01_solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
vanilla_gae = _mod.vanilla_gae


class TestVanillaGAE:
    def test_single_step(self):
        """With a single timestep, advantage = reward - value (next_value=0)."""
        rewards = np.array([[5.0]])
        values = np.array([[2.0]])
        adv, ret = vanilla_gae(rewards, values, gamma=0.99, lambd=0.95)
        # delta = 5 + 0.99*0 - 2 = 3
        np.testing.assert_allclose(adv, [[3.0]])
        np.testing.assert_allclose(ret, [[5.0]])  # 3 + 2

    def test_two_steps_hand_computed(self):
        """Hand-computed GAE for two timesteps."""
        rewards = np.array([[1.0, 2.0]])
        values = np.array([[0.5, 1.0]])
        gamma, lambd = 0.99, 0.95

        # delta_1 = 2.0 + 0.99*0.0 - 1.0 = 1.0
        # A_1 = delta_1 = 1.0
        # delta_0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
        # A_0 = delta_0 + 0.99*0.95*A_1 = 1.49 + 0.9405*1.0 = 2.4305
        adv, ret = vanilla_gae(rewards, values, gamma, lambd)
        np.testing.assert_allclose(adv[0, 1], 1.0, atol=1e-7)
        np.testing.assert_allclose(adv[0, 0], 2.4305, atol=1e-7)
        np.testing.assert_allclose(ret, adv + values, atol=1e-7)

    def test_zero_lambda_reduces_to_td(self):
        """When lambda=0, GAE reduces to one-step TD error."""
        rewards = np.array([[1.0, 2.0, 3.0]])
        values = np.array([[0.5, 1.5, 2.5]])
        gamma = 0.99

        adv, ret = vanilla_gae(rewards, values, gamma, lambd=0.0)
        # delta_0 = 1.0 + 0.99*1.5 - 0.5 = 1.985
        # delta_1 = 2.0 + 0.99*2.5 - 1.5 = 2.975
        # delta_2 = 3.0 + 0.0 - 2.5 = 0.5
        np.testing.assert_allclose(adv[0, 0], 1.985, atol=1e-7)
        np.testing.assert_allclose(adv[0, 1], 2.975, atol=1e-7)
        np.testing.assert_allclose(adv[0, 2], 0.5, atol=1e-7)

    def test_batch_dimension(self):
        """Verify GAE works correctly on batched input."""
        rewards = np.array([[1.0, 0.0], [0.0, 1.0]])
        values = np.array([[0.0, 0.0], [0.0, 0.0]])
        gamma, lambd = 1.0, 1.0

        adv, ret = vanilla_gae(rewards, values, gamma, lambd)
        # Batch 0: delta=[1,0], A_1=0, A_0=1+1*1*0=1
        np.testing.assert_allclose(adv[0], [1.0, 0.0], atol=1e-7)
        # Batch 1: delta=[0,1], A_1=1, A_0=0+1*1*1=1
        np.testing.assert_allclose(adv[1], [1.0, 1.0], atol=1e-7)

    def test_returns_equal_advantages_plus_values(self):
        """Returns should always be advantages + values."""
        rng = np.random.default_rng(42)
        rewards = rng.standard_normal((3, 10))
        values = rng.standard_normal((3, 10))
        adv, ret = vanilla_gae(rewards, values, gamma=0.99, lambd=0.95)
        np.testing.assert_allclose(ret, adv + values, atol=1e-10)

    def test_gamma_zero(self):
        """With gamma=0, advantage = reward - value at each step (no future)."""
        rewards = np.array([[1.0, 2.0, 3.0]])
        values = np.array([[0.5, 1.0, 1.5]])
        adv, ret = vanilla_gae(rewards, values, gamma=0.0, lambd=0.95)
        expected = rewards - values
        np.testing.assert_allclose(adv, expected, atol=1e-7)
