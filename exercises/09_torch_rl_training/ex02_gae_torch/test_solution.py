"""Tests for Exercise 02: GAE in PyTorch"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_gae = _mod.compute_gae


def test_basic_shape():
    """Output shapes should match input shapes."""
    B, T = 4, 10
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)

    adv, ret = compute_gae(rewards, values, gamma=0.99, lambd=0.95)
    assert adv.shape == (B, T)
    assert ret.shape == (B, T)


def test_returns_equal_advantages_plus_values():
    """returns = advantages + values by definition."""
    B, T = 3, 8
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)

    adv, ret = compute_gae(rewards, values, gamma=0.99, lambd=0.95)
    assert torch.allclose(ret, adv + values, atol=1e-5)


def test_single_step():
    """With T=1, advantage = delta = reward - value (since V_{T}=0)."""
    rewards = torch.tensor([[5.0]])
    values = torch.tensor([[2.0]])

    adv, ret = compute_gae(rewards, values, gamma=0.99, lambd=0.95)
    # delta = 5.0 + 0.99*0 - 2.0 = 3.0
    assert torch.isclose(adv[0, 0], torch.tensor(3.0), atol=1e-5)
    assert torch.isclose(ret[0, 0], torch.tensor(5.0), atol=1e-5)


def test_two_steps_known_values():
    """Hand-computed GAE for T=2."""
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 1.0]])
    gamma, lambd = 0.99, 0.95

    adv, ret = compute_gae(rewards, values, gamma=gamma, lambd=lambd)

    # delta_1 = 2.0 + 0.99*0 - 1.0 = 1.0
    # delta_0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
    # A_1 = delta_1 = 1.0
    # A_0 = delta_0 + gamma*lambd*A_1 = 1.49 + 0.99*0.95*1.0 = 1.49 + 0.9405 = 2.4305
    expected_a0 = 2.4305
    expected_a1 = 1.0

    assert torch.isclose(adv[0, 0], torch.tensor(expected_a0), atol=1e-4)
    assert torch.isclose(adv[0, 1], torch.tensor(expected_a1), atol=1e-4)


def test_lambda_zero():
    """With lambda=0, GAE reduces to TD(0): A_t = delta_t."""
    B, T = 2, 5
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    gamma = 0.99

    adv, _ = compute_gae(rewards, values, gamma=gamma, lambd=0.0)

    # TD residuals
    next_v = torch.cat([values[:, 1:], torch.zeros(B, 1)], dim=1)
    deltas = rewards + gamma * next_v - values
    assert torch.allclose(adv, deltas, atol=1e-5)


def test_lambda_one():
    """With lambda=1, GAE = discounted returns - values (MC estimate)."""
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    values = torch.zeros(1, 4)
    gamma = 1.0

    adv, _ = compute_gae(rewards, values, gamma=gamma, lambd=1.0)

    # With gamma=1, lambda=1, values=0: A_t = sum of future rewards
    expected = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    assert torch.allclose(adv, expected, atol=1e-5)


def test_batch_independent():
    """Each batch element should be computed independently."""
    rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.zeros(2, 2)

    adv, _ = compute_gae(rewards, values, gamma=1.0, lambd=1.0)

    # Batch 0: A_0 = 1+0=1, A_1 = 0
    # Batch 1: A_0 = 0+1=1, A_1 = 1
    assert torch.isclose(adv[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(adv[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(adv[1, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(adv[1, 1], torch.tensor(1.0), atol=1e-5)
