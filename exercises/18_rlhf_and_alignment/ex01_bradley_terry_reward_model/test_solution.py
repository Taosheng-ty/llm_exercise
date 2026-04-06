"""Tests for Exercise 01: Bradley-Terry Reward Model"""

import importlib.util
import os
import torch
import torch.nn.functional as F
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
reward_model_loss = _mod.reward_model_loss
compute_reward_margins = _mod.compute_reward_margins


class TestRewardModelLoss:
    def test_perfect_ranking_low_loss(self):
        """When chosen always scores much higher, loss should be near zero."""
        chosen = torch.tensor([5.0, 4.0, 6.0])
        rejected = torch.tensor([-5.0, -4.0, -6.0])
        loss, acc = reward_model_loss(chosen, rejected)
        assert loss.item() < 0.01
        assert acc == pytest.approx(1.0)

    def test_inverted_ranking_high_loss(self):
        """When rejected scores higher than chosen, loss should be large."""
        chosen = torch.tensor([-5.0, -4.0])
        rejected = torch.tensor([5.0, 4.0])
        loss, acc = reward_model_loss(chosen, rejected)
        assert loss.item() > 5.0
        assert acc == pytest.approx(0.0)

    def test_equal_rewards(self):
        """When rewards are equal, loss = -log(sigmoid(0)) = log(2) ~ 0.693."""
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([1.0, 2.0, 3.0])
        loss, acc = reward_model_loss(chosen, rejected)
        assert loss.item() == pytest.approx(0.6931, abs=1e-3)
        assert acc == pytest.approx(0.0)

    def test_accuracy_mixed_batch(self):
        """Accuracy should correctly reflect fraction of correct rankings."""
        chosen = torch.tensor([3.0, 1.0, 5.0, 0.5])
        rejected = torch.tensor([1.0, 3.0, 2.0, 0.5])
        _, acc = reward_model_loss(chosen, rejected)
        # 2 correct (3>1, 5>2), 1 wrong (1<3), 1 tie (0.5==0.5)
        assert acc == pytest.approx(0.5)

    def test_loss_is_scalar(self):
        """Loss should be a 0-dim tensor."""
        loss, _ = reward_model_loss(torch.randn(5), torch.randn(5))
        assert loss.dim() == 0

    def test_gradient_flows_through_loss(self):
        """Loss must be differentiable w.r.t. both inputs."""
        chosen = torch.randn(4, requires_grad=True)
        rejected = torch.randn(4, requires_grad=True)
        loss, _ = reward_model_loss(chosen, rejected)
        loss.backward()
        assert chosen.grad is not None
        assert rejected.grad is not None

    def test_gradient_direction(self):
        """Gradient should push chosen rewards up and rejected rewards down."""
        chosen = torch.tensor([0.0], requires_grad=True)
        rejected = torch.tensor([0.0], requires_grad=True)
        loss, _ = reward_model_loss(chosen, rejected)
        loss.backward()
        # d(loss)/d(chosen) should be negative (want to increase chosen)
        assert chosen.grad.item() < 0
        # d(loss)/d(rejected) should be positive (want to decrease rejected)
        assert rejected.grad.item() > 0

    def test_batch_size_one(self):
        """Should work with a single pair."""
        loss, acc = reward_model_loss(torch.tensor([2.0]), torch.tensor([1.0]))
        assert loss.item() > 0
        assert acc == pytest.approx(1.0)

    def test_large_batch(self):
        """Should handle large batches without issues."""
        B = 1024
        chosen = torch.randn(B) + 1.0  # slight positive bias
        rejected = torch.randn(B) - 1.0
        loss, acc = reward_model_loss(chosen, rejected)
        assert loss.item() > 0
        assert 0.0 <= acc <= 1.0


class TestComputeRewardMargins:
    def test_basic_margins(self):
        """Margins should be the element-wise difference."""
        chosen = torch.tensor([3.0, 5.0, 1.0])
        rejected = torch.tensor([1.0, 2.0, 4.0])
        margins = compute_reward_margins(chosen, rejected)
        expected = torch.tensor([2.0, 3.0, -3.0])
        assert torch.allclose(margins, expected)

    def test_margins_shape(self):
        """Output shape should match input shape."""
        B = 7
        margins = compute_reward_margins(torch.randn(B), torch.randn(B))
        assert margins.shape == (B,)

    def test_zero_margins_for_equal_rewards(self):
        """Equal rewards should produce zero margins."""
        vals = torch.tensor([1.0, 2.0, 3.0])
        margins = compute_reward_margins(vals, vals)
        assert torch.allclose(margins, torch.zeros(3))
