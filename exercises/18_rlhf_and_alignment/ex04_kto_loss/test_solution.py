"""Tests for Exercise 04: Kahneman-Tversky Optimization (KTO) Loss"""

import importlib.util
import os
import torch
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
kto_loss = _mod.kto_loss


class TestKTOLoss:
    def test_desirable_high_logratio_low_loss(self):
        """Desirable sample with high log-ratio should have low loss."""
        policy_logps = torch.tensor([0.0])   # policy likes it
        ref_logps = torch.tensor([-5.0])     # ref doesn't
        is_desirable = torch.tensor([True])
        loss, metrics = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        # log_ratio = 5.0, sigmoid(1.0 * (5.0 - 0.0)) ~ 1.0, loss ~ 0
        assert loss.item() < 0.01

    def test_desirable_low_logratio_high_loss(self):
        """Desirable sample with low log-ratio should have high loss."""
        policy_logps = torch.tensor([-5.0])
        ref_logps = torch.tensor([0.0])
        is_desirable = torch.tensor([True])
        loss, _ = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        # log_ratio = -5.0, sigmoid(-5.0) ~ 0.007, loss ~ 0.993
        assert loss.item() > 0.9

    def test_undesirable_low_logratio_low_loss(self):
        """Undesirable sample with low log-ratio should have low loss."""
        policy_logps = torch.tensor([-5.0])
        ref_logps = torch.tensor([0.0])
        is_desirable = torch.tensor([False])
        loss, _ = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        # log_ratio = -5.0, sigmoid(1.0 * (0.0 - (-5.0))) = sigmoid(5.0) ~ 1.0, loss ~ 0
        assert loss.item() < 0.01

    def test_undesirable_high_logratio_high_loss(self):
        """Undesirable sample with high log-ratio should have high loss."""
        policy_logps = torch.tensor([0.0])
        ref_logps = torch.tensor([-5.0])
        is_desirable = torch.tensor([False])
        loss, _ = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        # log_ratio = 5.0, sigmoid(0.0 - 5.0) = sigmoid(-5) ~ 0.007, loss ~ 0.993
        assert loss.item() > 0.9

    def test_mixed_batch(self):
        """Should handle mixed desirable/undesirable samples."""
        policy_logps = torch.tensor([0.0, -5.0, 0.0, -5.0])
        ref_logps = torch.tensor([-5.0, 0.0, -5.0, 0.0])
        is_desirable = torch.tensor([True, True, False, False])
        loss, metrics = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        # Sample 0: desirable, high lr -> low loss
        # Sample 1: desirable, low lr -> high loss
        # Sample 2: undesirable, high lr -> high loss
        # Sample 3: undesirable, low lr -> low loss
        assert 0.0 < loss.item() < 1.0

    def test_all_desirable(self):
        """Should handle batch with only desirable samples."""
        policy_logps = torch.tensor([1.0, 2.0])
        ref_logps = torch.tensor([0.0, 0.0])
        is_desirable = torch.tensor([True, True])
        loss, metrics = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=0.1)
        assert metrics["undesirable_loss"] == 0.0
        assert metrics["desirable_loss"] > 0.0

    def test_all_undesirable(self):
        """Should handle batch with only undesirable samples."""
        policy_logps = torch.tensor([1.0, 2.0])
        ref_logps = torch.tensor([0.0, 0.0])
        is_desirable = torch.tensor([False, False])
        loss, metrics = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=0.1)
        assert metrics["desirable_loss"] == 0.0
        assert metrics["undesirable_loss"] > 0.0

    def test_metrics_keys(self):
        """Metrics should contain expected keys."""
        _, metrics = kto_loss(
            torch.randn(3), torch.randn(3),
            torch.tensor([True, False, True]), 0.5, beta=0.1,
        )
        assert "log_ratios" in metrics
        assert "desirable_loss" in metrics
        assert "undesirable_loss" in metrics

    def test_log_ratios_shape(self):
        """Log ratios in metrics should have batch shape."""
        B = 4
        _, metrics = kto_loss(
            torch.randn(B), torch.randn(B),
            torch.ones(B, dtype=torch.bool), 0.0, beta=0.1,
        )
        assert metrics["log_ratios"].shape == (B,)

    def test_loss_is_scalar(self):
        """Loss should be a 0-dim tensor."""
        loss, _ = kto_loss(
            torch.randn(5), torch.randn(5),
            torch.ones(5, dtype=torch.bool), 0.0,
        )
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """Loss should be differentiable w.r.t. policy log probs."""
        policy_logps = torch.randn(4, requires_grad=True)
        ref_logps = torch.randn(4)
        is_desirable = torch.tensor([True, False, True, False])
        loss, _ = kto_loss(policy_logps, ref_logps, is_desirable, 0.0, beta=0.5)
        loss.backward()
        assert policy_logps.grad is not None

    def test_kl_estimate_shifts_behavior(self):
        """Changing kl_estimate should shift where the loss is minimized."""
        policy_logps = torch.tensor([2.0])
        ref_logps = torch.tensor([0.0])
        is_desirable = torch.tensor([True])
        loss_kl0, _ = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=0.0, beta=1.0)
        loss_kl5, _ = kto_loss(policy_logps, ref_logps, is_desirable, kl_estimate=5.0, beta=1.0)
        # With higher kl_estimate, the effective argument to sigmoid is smaller,
        # so loss is higher
        assert loss_kl5.item() > loss_kl0.item()
