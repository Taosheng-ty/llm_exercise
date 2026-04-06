"""Tests for Exercise 01: AdamW Optimizer from Scratch"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AdamW = _mod.AdamW


def _make_simple_model():
    """Create a simple linear model for testing."""
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2, bias=True)
    return model


class TestAdamWInit:
    def test_params_stored(self):
        """Optimizer should store parameters."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.01)
        assert len(opt.params) == 2  # weight + bias

    def test_state_initialized(self):
        """Optimizer state should have m and v initialized to zeros."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.01)
        for p in opt.params:
            assert torch.all(opt.state[p]["m"] == 0)
            assert torch.all(opt.state[p]["v"] == 0)
            assert opt.state[p]["step"] == 0


class TestAdamWStep:
    def test_single_step_changes_params(self):
        """After one step, parameters should change."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.01)
        old_weight = model.weight.data.clone()
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        assert not torch.equal(model.weight.data, old_weight)

    def test_zero_grad(self):
        """zero_grad should clear gradients."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.01)
        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None
        opt.zero_grad()
        assert model.weight.grad is None

    def test_step_count_increments(self):
        """Step counter should increment with each step call."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.01)
        for i in range(5):
            x = torch.randn(3, 4)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        for p in opt.params:
            assert opt.state[p]["step"] == 5

    def test_matches_torch_adamw_no_weight_decay(self):
        """With weight_decay=0, should match torch AdamW closely."""
        torch.manual_seed(0)
        model1 = torch.nn.Linear(4, 2, bias=False)
        torch.manual_seed(0)
        model2 = torch.nn.Linear(4, 2, bias=False)

        opt1 = AdamW(model1.parameters(), lr=0.01, weight_decay=0.0)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01, weight_decay=0.0)

        for _ in range(10):
            torch.manual_seed(_)
            x = torch.randn(3, 4)

            loss1 = model1(x).sum()
            loss1.backward()
            opt1.step()
            opt1.zero_grad()

            loss2 = model2(x).sum()
            loss2.backward()
            opt2.step()
            opt2.zero_grad()

        torch.testing.assert_close(model1.weight.data, model2.weight.data, atol=1e-5, rtol=1e-5)

    def test_matches_torch_adamw_with_weight_decay(self):
        """With weight_decay>0, should match torch AdamW closely."""
        torch.manual_seed(0)
        model1 = torch.nn.Linear(4, 2, bias=False)
        torch.manual_seed(0)
        model2 = torch.nn.Linear(4, 2, bias=False)

        opt1 = AdamW(model1.parameters(), lr=0.01, weight_decay=0.1)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01, weight_decay=0.1)

        for _ in range(10):
            torch.manual_seed(_)
            x = torch.randn(3, 4)

            loss1 = model1(x).sum()
            loss1.backward()
            opt1.step()
            opt1.zero_grad()

            loss2 = model2(x).sum()
            loss2.backward()
            opt2.step()
            opt2.zero_grad()

        torch.testing.assert_close(model1.weight.data, model2.weight.data, atol=1e-5, rtol=1e-5)

    def test_weight_decay_is_decoupled(self):
        """Decoupled weight decay should shrink params toward zero independently of gradients."""
        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.tensor([10.0]))
        opt = AdamW([p], lr=0.0, weight_decay=0.1)
        # With lr=0 and weight_decay=0.1, the param should NOT change
        # because decoupled WD is multiplied by lr: param -= lr * (... + wd * param)
        old_val = p.data.clone()
        p.grad = torch.tensor([0.0])
        opt.step()
        # lr=0 means no update at all
        torch.testing.assert_close(p.data, old_val)

    def test_momentum_accumulates(self):
        """First moment should accumulate gradient information over steps."""
        model = _make_simple_model()
        opt = AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        for _ in range(3):
            x = torch.ones(1, 4)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        # m should be non-zero after training
        for p in opt.params:
            assert not torch.all(opt.state[p]["m"] == 0)

    def test_no_grad_param_skipped(self):
        """Parameters without gradients should be skipped."""
        p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        p2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        opt = AdamW([p1, p2], lr=0.01)
        # Only set grad for p1
        p1.grad = torch.tensor([1.0, 1.0])
        old_p2 = p2.data.clone()
        opt.step()
        torch.testing.assert_close(p2.data, old_p2)
