"""
Tests for Exercise 05: Multi-Token Prediction
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultiTokenPredictionHead = _mod.MultiTokenPredictionHead
compute_multi_token_loss = _mod.compute_multi_token_loss


class TestMultiTokenPredictionHead:
    def test_is_nn_module(self):
        head = MultiTokenPredictionHead(d_model=64, vocab_size=100, num_futures=3)
        assert isinstance(head, nn.Module)

    def test_output_is_list_of_correct_length(self):
        head = MultiTokenPredictionHead(d_model=64, vocab_size=100, num_futures=4)
        hidden = torch.randn(2, 10, 64)
        preds = head(hidden)
        assert isinstance(preds, list)
        assert len(preds) == 4

    def test_output_shapes(self):
        B, S, D, V, N = 2, 8, 32, 50, 3
        head = MultiTokenPredictionHead(d_model=D, vocab_size=V, num_futures=N)
        hidden = torch.randn(B, S, D)
        preds = head(hidden)
        for pred in preds:
            assert pred.shape == (B, S, V)

    def test_heads_are_independent(self):
        """Different heads should produce different predictions."""
        torch.manual_seed(42)
        head = MultiTokenPredictionHead(d_model=32, vocab_size=50, num_futures=3)
        hidden = torch.randn(1, 5, 32)
        preds = head(hidden)
        # With random init, heads should give different outputs
        assert not torch.allclose(preds[0], preds[1], atol=1e-3)

    def test_gradients_flow_to_all_heads(self):
        head = MultiTokenPredictionHead(d_model=32, vocab_size=50, num_futures=3)
        hidden = torch.randn(2, 5, 32)
        preds = head(hidden)
        loss = sum(p.sum() for p in preds)
        loss.backward()
        # All head parameters should have gradients
        for i, h in enumerate(head.heads):
            for p in h.parameters():
                assert p.grad is not None, f"Head {i} has no gradient"
                assert p.grad.abs().sum() > 0

    def test_num_parameters(self):
        D, V, N = 64, 100, 3
        head = MultiTokenPredictionHead(d_model=D, vocab_size=V, num_futures=N)
        total = sum(p.numel() for p in head.parameters())
        # N linear layers: each has D*V weights + V bias
        expected = N * (D * V + V)
        assert total == expected


class TestComputeMultiTokenLoss:
    def test_returns_tuple(self):
        preds = [torch.randn(2, 8, 50) for _ in range(3)]
        targets = torch.randint(0, 50, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)
        total, per_head = compute_multi_token_loss(preds, targets, mask)
        assert isinstance(total, torch.Tensor)
        assert total.ndim == 0  # scalar
        assert isinstance(per_head, list)
        assert len(per_head) == 3

    def test_loss_is_positive(self):
        torch.manual_seed(42)
        preds = [torch.randn(2, 10, 50) for _ in range(3)]
        targets = torch.randint(0, 50, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.bool)
        total, per_head = compute_multi_token_loss(preds, targets, mask)
        assert total.item() > 0
        for h_loss in per_head:
            assert h_loss.item() > 0

    def test_per_head_losses_are_individual(self):
        """Each head loss should be independently computed."""
        torch.manual_seed(0)
        preds = [torch.randn(2, 10, 50) for _ in range(3)]
        targets = torch.randint(0, 50, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.bool)
        _, per_head = compute_multi_token_loss(preds, targets, mask)
        # With random predictions, head losses should differ
        assert not (per_head[0].item() == per_head[1].item() == per_head[2].item())

    def test_mask_zeros_exclude_positions(self):
        """Masked positions should not contribute to loss."""
        torch.manual_seed(42)
        preds = [torch.randn(1, 8, 20) for _ in range(2)]
        targets = torch.randint(0, 20, (1, 8))
        mask_all = torch.ones(1, 8, dtype=torch.bool)
        mask_half = torch.tensor([[True, True, True, True, False, False, False, False]])
        total_all, _ = compute_multi_token_loss(preds, targets, mask_all)
        total_half, _ = compute_multi_token_loss(preds, targets, mask_half)
        # Different masks should give different losses
        assert total_all.item() != total_half.item()

    def test_correct_target_shifting(self):
        """Head i should predict targets shifted by i+1 positions."""
        B, S, V = 1, 6, 10
        # Create predictions that are one-hot for specific targets
        targets = torch.tensor([[0, 1, 2, 3, 4, 5]])
        preds = []
        for i in range(2):
            # Create logits that perfectly predict the shifted targets
            logits = torch.full((B, S, V), -100.0)
            shift = i + 1
            for t in range(S - shift):
                target_token = targets[0, t + shift].item()
                logits[0, t, target_token] = 100.0
            preds.append(logits)
        mask = torch.ones(B, S, dtype=torch.bool)
        total, per_head = compute_multi_token_loss(preds, targets, mask)
        # Loss should be very close to 0 since predictions match targets
        assert total.item() < 0.01

    def test_loss_gradient_flows(self):
        """Loss should be differentiable."""
        preds = [torch.randn(2, 8, 30, requires_grad=True) for _ in range(3)]
        targets = torch.randint(0, 30, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.bool)
        total, _ = compute_multi_token_loss(preds, targets, mask)
        total.backward()
        for p in preds:
            assert p.grad is not None
