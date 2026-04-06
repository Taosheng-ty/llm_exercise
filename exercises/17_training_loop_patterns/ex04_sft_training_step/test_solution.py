"""Tests for Exercise 04: SFT Training Step"""

import importlib.util
import os
import torch
import torch.nn as nn
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_sft_loss = _mod.compute_sft_loss
sft_training_step = _mod.sft_training_step


class TinyLM(nn.Module):
    """Minimal language model for testing."""

    def __init__(self, vocab_size=32, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


class TestComputeSFTLoss:
    def test_all_masked_out(self):
        """All-zero mask should return 0 loss (or near-zero)."""
        logits = torch.randn(2, 5, 10)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.zeros(2, 5)
        loss = compute_sft_loss(logits, labels, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_perfect_prediction(self):
        """When logits strongly predict correct labels, loss should be near 0."""
        B, T, V = 1, 4, 8
        labels = torch.tensor([[0, 1, 2, 3]])
        # Create logits that strongly predict the next token
        logits = torch.full((B, T, V), -10.0)
        for t in range(T - 1):
            logits[0, t, labels[0, t + 1]] = 10.0
        mask = torch.ones(B, T)
        loss = compute_sft_loss(logits, labels, mask)
        assert loss.item() < 0.01

    def test_mask_only_last_tokens(self):
        """Only response tokens (masked=1) should contribute to loss."""
        B, T, V = 1, 6, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        # Mask: only last 2 tokens are response
        mask = torch.tensor([[0, 0, 0, 0, 1, 1]], dtype=torch.float)
        loss_masked = compute_sft_loss(logits, labels, mask)

        # Compare with manual computation on those positions
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = mask[:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), reduction="none"
        ).view(B, T - 1)
        expected = (per_token * shift_mask).sum() / shift_mask.sum()
        assert loss_masked.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_loss_is_scalar(self):
        logits = torch.randn(2, 5, 10)
        labels = torch.randint(0, 10, (2, 5))
        mask = torch.ones(2, 5)
        loss = compute_sft_loss(logits, labels, mask)
        assert loss.dim() == 0


class TestSFTTrainingStep:
    def test_loss_decreases(self):
        """Loss should decrease over multiple training steps."""
        torch.manual_seed(42)
        model = TinyLM(vocab_size=16, hidden_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        input_ids = torch.randint(0, 16, (4, 10))
        mask = torch.ones(4, 10)

        losses = []
        for _ in range(20):
            loss = sft_training_step(model, input_ids, mask, optimizer)
            losses.append(loss)

        assert losses[-1] < losses[0], "Loss should decrease over training"

    def test_returns_float(self):
        torch.manual_seed(0)
        model = TinyLM()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        ids = torch.randint(0, 32, (2, 5))
        mask = torch.ones(2, 5)
        loss = sft_training_step(model, ids, mask, opt)
        assert isinstance(loss, float)

    def test_parameters_updated(self):
        """Model parameters should change after a training step."""
        torch.manual_seed(0)
        model = TinyLM()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        ids = torch.randint(0, 32, (2, 8))
        mask = torch.ones(2, 8)
        old_params = [p.clone() for p in model.parameters()]
        sft_training_step(model, ids, mask, opt)
        changed = any(
            not torch.equal(old, new)
            for old, new in zip(old_params, model.parameters())
        )
        assert changed, "Parameters should change after training step"
