"""Tests for Exercise 03: Clipped Value Function Loss"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_value_loss = _mod.compute_value_loss


def test_basic_shape():
    """Loss and clip_frac should be scalars."""
    B, T = 3, 8
    values = torch.randn(B, T, requires_grad=True)
    old_values = torch.randn(B, T)
    returns = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_value_loss(values, old_values, returns, mask)
    assert loss.dim() == 0
    assert clip_frac.dim() == 0


def test_differentiable():
    """Loss should be differentiable w.r.t. values."""
    B, T = 2, 5
    values = torch.randn(B, T, requires_grad=True)
    old_values = torch.randn(B, T)
    returns = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, _ = compute_value_loss(values, old_values, returns, mask)
    loss.backward()
    assert values.grad is not None
    assert values.grad.shape == values.shape


def test_no_clip_when_values_close():
    """No clipping when values == old_values."""
    B, T = 2, 4
    old_values = torch.randn(B, T)
    values = old_values.clone().detach().requires_grad_(True)
    returns = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_value_loss(values, old_values, returns, mask)
    assert torch.isclose(clip_frac, torch.tensor(0.0), atol=1e-6)


def test_clipping_happens():
    """Large value change should trigger clipping."""
    B, T = 1, 4
    old_values = torch.zeros(B, T)
    # values differ by 5.0 >> value_clip=0.2
    values = (5.0 * torch.ones(B, T)).requires_grad_(True)
    returns = torch.zeros(B, T)
    mask = torch.ones(B, T)

    loss, clip_frac = compute_value_loss(values, old_values, returns, mask, value_clip=0.2)
    assert clip_frac.item() == 1.0, f"Expected all tokens clipped, got {clip_frac.item()}"


def test_mask_respected():
    """Masked positions should not affect loss."""
    B, T = 1, 4
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    old_values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    returns = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss, _ = compute_value_loss(values, old_values, returns, mask, value_clip=10.0)

    # Only first two tokens: (1-0)^2 + (2-0)^2 = 1 + 4 = 5, mean = 2.5
    expected = torch.tensor(2.5)
    assert torch.isclose(loss, expected, atol=1e-5), f"Loss {loss.item()} != expected {expected.item()}"


def test_known_values():
    """Hand-computed test case."""
    old_values = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[1.5, 2.5]], requires_grad=True)
    returns = torch.tensor([[1.2, 2.3]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, clip_frac = compute_value_loss(old_values=old_values, values=values, returns=returns,
                                          loss_mask=mask, value_clip=0.2)

    # values_clipped = [1+clamp(0.5,-0.2,0.2), 2+clamp(0.5,-0.2,0.2)] = [1.2, 2.2]
    # surr1 = [(1.2-1.2)^2, (2.2-2.3)^2] = [0.0, 0.01]
    # surr2 = [(1.5-1.2)^2, (2.5-2.3)^2] = [0.09, 0.04]
    # max = [0.09, 0.04], mean = 0.065
    expected_loss = torch.tensor(0.065)
    assert torch.isclose(loss, expected_loss, atol=1e-4), f"Loss {loss.item()} != expected {expected_loss.item()}"
    assert clip_frac.item() == 1.0  # Both tokens exceed clip threshold
