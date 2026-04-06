"""Tests for Exercise 02: Gradient Clipping"""

import importlib.util
import os
import torch
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
clip_grad_norm = _mod.clip_grad_norm


class TestGradientClipping:
    def test_no_clipping_needed(self):
        """When norm < max_norm, gradients should not change."""
        model = torch.nn.Linear(3, 2, bias=False)
        model.weight.grad = torch.ones(2, 3) * 0.1  # norm ~ 0.245
        orig_grad = model.weight.grad.clone()
        norm = clip_grad_norm(model.parameters(), max_norm=10.0)
        torch.testing.assert_close(model.weight.grad, orig_grad)
        assert norm < 10.0

    def test_clipping_scales_gradients(self):
        """When norm > max_norm, gradients should be scaled down."""
        model = torch.nn.Linear(3, 2, bias=False)
        model.weight.grad = torch.ones(2, 3) * 10.0  # norm = sqrt(6)*10 ~ 24.5
        norm = clip_grad_norm(model.parameters(), max_norm=1.0)
        assert norm == pytest.approx((6 ** 0.5) * 10.0, rel=1e-5)
        # After clipping, norm should be max_norm
        new_norm = model.weight.grad.norm(2).item()
        assert new_norm == pytest.approx(1.0, rel=1e-5)

    def test_returns_original_norm(self):
        """Should return the pre-clipping norm."""
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])  # norm = 5
        norm = clip_grad_norm([p], max_norm=1.0)
        assert norm == pytest.approx(5.0, rel=1e-5)

    def test_multiple_parameters(self):
        """Global norm should be computed across all parameters."""
        p1 = torch.nn.Parameter(torch.zeros(3))
        p1.grad = torch.tensor([3.0, 0.0, 0.0])  # norm = 3
        p2 = torch.nn.Parameter(torch.zeros(2))
        p2.grad = torch.tensor([0.0, 4.0])  # norm = 4
        # global norm = sqrt(9 + 16) = 5
        norm = clip_grad_norm([p1, p2], max_norm=2.5)
        assert norm == pytest.approx(5.0, rel=1e-5)
        new_norm = (p1.grad.norm(2) ** 2 + p2.grad.norm(2) ** 2) ** 0.5
        assert new_norm.item() == pytest.approx(2.5, rel=1e-5)

    def test_none_grad_skipped(self):
        """Parameters with None grad should be skipped."""
        p1 = torch.nn.Parameter(torch.zeros(3))
        p1.grad = torch.tensor([1.0, 0.0, 0.0])
        p2 = torch.nn.Parameter(torch.zeros(2))
        p2.grad = None
        norm = clip_grad_norm([p1, p2], max_norm=10.0)
        assert norm == pytest.approx(1.0, rel=1e-5)

    def test_exact_max_norm(self):
        """When norm == max_norm, no clipping should happen."""
        p = torch.nn.Parameter(torch.zeros(1))
        p.grad = torch.tensor([5.0])
        orig = p.grad.clone()
        norm = clip_grad_norm([p], max_norm=5.0)
        assert norm == pytest.approx(5.0)
        torch.testing.assert_close(p.grad, orig)

    def test_empty_parameters(self):
        """No parameters should return 0 norm."""
        norm = clip_grad_norm([], max_norm=1.0)
        assert norm == 0.0
