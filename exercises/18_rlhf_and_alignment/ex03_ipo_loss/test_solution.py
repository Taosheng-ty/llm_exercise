"""Tests for Exercise 03: Identity Preference Optimization (IPO) Loss"""

import importlib.util
import os
import torch
import torch.nn.functional as F
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ipo_loss = _mod.ipo_loss


class TestIPOLoss:
    def test_loss_at_target_margin_is_zero(self):
        """When margin exactly equals 1/(2*beta), loss should be zero."""
        beta = 0.5
        target = 1.0 / (2.0 * beta)  # = 1.0
        # Set up so that margin = target
        # log_ratio_chosen - log_ratio_rejected = 1.0
        # (pc - rc) - (pr - rr) = 1.0
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-3.0])
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.0])
        # margin = (-1 - (-2)) - (-3 - (-2)) = 1 - (-1) = 2... not 1
        # Let's be precise: margin = (pc-rc) - (pr-rr) = (-1+2) - (-3+2) = 1 - (-1) = 2
        # Need margin = 1.0, so adjust:
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-2.0])
        rc = torch.tensor([-1.0])
        rr = torch.tensor([-1.0])
        # margin = (-1+1) - (-2+1) = 0 - (-1) = 1.0 = target
        loss, metrics = ipo_loss(pc, pr, rc, rr, beta=beta)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_increases_away_from_target(self):
        """Loss should increase as margin deviates from target."""
        beta = 0.1
        rc = torch.tensor([-2.0])
        rr = torch.tensor([-2.0])

        # Margin close to target
        pc_close = torch.tensor([-2.0 + 5.0])  # log_ratio_chosen = 5.0
        pr_close = torch.tensor([-2.0])          # log_ratio_rejected = 0.0
        # margin = 5.0, target = 5.0
        loss_close, _ = ipo_loss(pc_close, pr_close, rc, rr, beta=beta)

        # Margin far from target
        pc_far = torch.tensor([-2.0 + 10.0])
        loss_far, _ = ipo_loss(pc_far, pr_close, rc, rr, beta=beta)

        assert loss_far.item() > loss_close.item()

    def test_metrics_keys(self):
        """Metrics should contain expected keys."""
        loss, metrics = ipo_loss(
            torch.randn(3), torch.randn(3), torch.randn(3), torch.randn(3)
        )
        assert "log_ratio_chosen" in metrics
        assert "log_ratio_rejected" in metrics
        assert "margin" in metrics
        assert "target" in metrics

    def test_metrics_shapes(self):
        """Metric tensors should have batch shape."""
        B = 5
        _, metrics = ipo_loss(
            torch.randn(B), torch.randn(B), torch.randn(B), torch.randn(B)
        )
        assert metrics["log_ratio_chosen"].shape == (B,)
        assert metrics["log_ratio_rejected"].shape == (B,)
        assert metrics["margin"].shape == (B,)

    def test_target_depends_on_beta(self):
        """Target margin should be 1/(2*beta)."""
        _, m1 = ipo_loss(torch.randn(2), torch.randn(2), torch.randn(2), torch.randn(2), beta=0.1)
        _, m2 = ipo_loss(torch.randn(2), torch.randn(2), torch.randn(2), torch.randn(2), beta=0.5)
        assert m1["target"] == pytest.approx(5.0)
        assert m2["target"] == pytest.approx(1.0)

    def test_loss_is_scalar(self):
        """Loss should be a 0-dim tensor."""
        loss, _ = ipo_loss(torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4))
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """Loss should be differentiable w.r.t. policy log probs."""
        pc = torch.randn(3, requires_grad=True)
        pr = torch.randn(3, requires_grad=True)
        rc = torch.randn(3)
        rr = torch.randn(3)
        loss, _ = ipo_loss(pc, pr, rc, rr)
        loss.backward()
        assert pc.grad is not None
        assert pr.grad is not None

    def test_loss_is_non_negative(self):
        """Squared loss should always be non-negative."""
        for _ in range(10):
            loss, _ = ipo_loss(
                torch.randn(8), torch.randn(8), torch.randn(8), torch.randn(8),
                beta=0.1 + torch.rand(1).item(),
            )
            assert loss.item() >= 0.0

    def test_ipo_vs_dpo_bounded_margin(self):
        """IPO loss should be low even for very large margins near the target,
        while having finite loss everywhere (unlike DPO which can saturate)."""
        beta = 0.1
        target = 1.0 / (2.0 * beta)  # 5.0
        # Margin at target => loss = 0
        pc = torch.tensor([target])
        pr = torch.tensor([0.0])
        rc = torch.tensor([0.0])
        rr = torch.tensor([0.0])
        loss_at_target, _ = ipo_loss(pc, pr, rc, rr, beta=beta)
        # Margin overshooting target by 10 => loss = 100
        pc_over = torch.tensor([target + 10.0])
        loss_over, _ = ipo_loss(pc_over, pr, rc, rr, beta=beta)
        assert loss_at_target.item() < loss_over.item()
        assert loss_over.item() == pytest.approx(100.0, abs=1e-4)

    def test_symmetric_penalty(self):
        """IPO should penalize equally for being above or below the target margin."""
        beta = 0.5
        target = 1.0  # 1/(2*0.5)
        rc = torch.tensor([0.0])
        rr = torch.tensor([0.0])
        # Margin = target + delta
        delta = 2.0
        loss_above, _ = ipo_loss(
            torch.tensor([target + delta]), torch.tensor([0.0]), rc, rr, beta=beta
        )
        loss_below, _ = ipo_loss(
            torch.tensor([target - delta]), torch.tensor([0.0]), rc, rr, beta=beta
        )
        assert loss_above.item() == pytest.approx(loss_below.item(), abs=1e-5)
