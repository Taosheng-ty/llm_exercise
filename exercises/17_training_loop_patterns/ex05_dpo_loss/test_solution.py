"""Tests for Exercise 05: Direct Preference Optimization (DPO) Loss"""

import importlib.util
import os
import torch
import torch.nn.functional as F
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_log_probs = _mod.compute_log_probs
dpo_loss = _mod.dpo_loss


class TestComputeLogProbs:
    def test_perfect_predictions(self):
        """Strong logits at correct positions should give high log probs."""
        B, T, V = 1, 4, 8
        labels = torch.tensor([[0, 1, 2, 3]])
        logits = torch.full((B, T, V), -10.0)
        for t in range(T - 1):
            logits[0, t, labels[0, t + 1]] = 10.0
        mask = torch.ones(B, T)
        lp = compute_log_probs(logits, labels, mask)
        # Should be close to 0 (high confidence)
        assert lp.item() > -0.1

    def test_mask_excludes_positions(self):
        """Masked-out positions should not contribute to log probs."""
        B, T, V = 1, 5, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        full_mask = torch.ones(B, T)
        partial_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float)

        lp_full = compute_log_probs(logits, labels, full_mask)
        lp_partial = compute_log_probs(logits, labels, partial_mask)
        # Partial should include fewer tokens -> different sum
        assert lp_full.item() != pytest.approx(lp_partial.item(), abs=1e-3)

    def test_output_shape(self):
        B, T, V = 3, 6, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T)
        lp = compute_log_probs(logits, labels, mask)
        assert lp.shape == (B,)

    def test_log_probs_are_negative(self):
        """Log probs should be non-positive."""
        B, T, V = 2, 5, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        mask = torch.ones(B, T)
        lp = compute_log_probs(logits, labels, mask)
        assert (lp <= 1e-6).all()


class TestDPOLoss:
    def test_chosen_preferred_gives_low_loss(self):
        """When policy prefers chosen more than ref does, loss should be low."""
        # Policy strongly prefers chosen
        policy_chosen = torch.tensor([-1.0])
        policy_rejected = torch.tensor([-5.0])
        # Reference is neutral
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])
        loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)
        assert loss.item() < 0.1

    def test_rejected_preferred_gives_high_loss(self):
        """When policy prefers rejected more than ref does, loss should be high."""
        policy_chosen = torch.tensor([-5.0])
        policy_rejected = torch.tensor([-1.0])
        ref_chosen = torch.tensor([-3.0])
        ref_rejected = torch.tensor([-3.0])
        loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)
        assert loss.item() > 1.0

    def test_metrics_accuracy(self):
        """Accuracy should reflect how often chosen_reward > rejected_reward."""
        policy_chosen = torch.tensor([-1.0, -5.0, -1.0, -5.0])
        policy_rejected = torch.tensor([-5.0, -1.0, -5.0, -1.0])
        ref_chosen = torch.tensor([-3.0, -3.0, -3.0, -3.0])
        ref_rejected = torch.tensor([-3.0, -3.0, -3.0, -3.0])
        _, metrics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0)
        # 2 out of 4 have chosen preferred
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_metrics_rewards_shape(self):
        B = 3
        policy_chosen = torch.randn(B)
        policy_rejected = torch.randn(B)
        ref_chosen = torch.randn(B)
        ref_rejected = torch.randn(B)
        _, metrics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        assert metrics["chosen_rewards"].shape == (B,)
        assert metrics["rejected_rewards"].shape == (B,)

    def test_beta_scaling(self):
        """Higher beta should amplify the reward differences."""
        pc = torch.tensor([-1.0])
        pr = torch.tensor([-2.0])
        rc = torch.tensor([-1.5])
        rr = torch.tensor([-1.5])
        _, m1 = dpo_loss(pc, pr, rc, rr, beta=0.1)
        _, m2 = dpo_loss(pc, pr, rc, rr, beta=1.0)
        # Rewards should scale with beta
        assert abs(m2["chosen_rewards"].item()) > abs(m1["chosen_rewards"].item())

    def test_loss_is_scalar(self):
        B = 4
        loss, _ = dpo_loss(torch.randn(B), torch.randn(B), torch.randn(B), torch.randn(B))
        assert loss.dim() == 0

    def test_gradient_flows(self):
        """Loss should be differentiable."""
        pc = torch.randn(3, requires_grad=True)
        pr = torch.randn(3, requires_grad=True)
        rc = torch.randn(3)
        rr = torch.randn(3)
        loss, _ = dpo_loss(pc, pr, rc, rr, beta=0.5)
        loss.backward()
        assert pc.grad is not None
        assert pr.grad is not None
