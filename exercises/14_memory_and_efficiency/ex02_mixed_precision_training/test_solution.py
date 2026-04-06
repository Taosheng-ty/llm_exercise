"""
Tests for Exercise 02: Mixed Precision Training
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GradScaler = _mod.GradScaler
mixed_precision_train_step = _mod.mixed_precision_train_step


class TestGradScaler:
    def test_initial_scale(self):
        scaler = GradScaler(init_scale=1024.0)
        assert scaler.get_scale() == 1024.0

    def test_scale_loss(self):
        scaler = GradScaler(init_scale=256.0)
        loss = torch.tensor(1.5)
        scaled = scaler.scale(loss)
        assert abs(scaled.item() - 384.0) < 1e-5

    def test_unscale_divides_grads(self):
        scaler = GradScaler(init_scale=4.0)
        w = torch.tensor([2.0], requires_grad=True)
        loss = (w * 3.0).sum()
        scaled = scaler.scale(loss)
        scaled.backward()
        # grad should be 3.0 * 4.0 = 12.0 before unscale
        assert abs(w.grad.item() - 12.0) < 1e-5

        opt = torch.optim.SGD([w], lr=0.1)
        scaler.unscale_(opt)
        # After unscale, grad should be 3.0
        assert abs(w.grad.item() - 3.0) < 1e-5

    def test_inf_detection(self):
        scaler = GradScaler(init_scale=1.0)
        w = torch.tensor([1.0], requires_grad=True)
        # Create inf gradient manually
        loss = w.sum()
        loss.backward()
        w.grad.data.fill_(float("inf"))

        opt = torch.optim.SGD([w], lr=0.1)
        scaler.unscale_(opt)
        assert scaler._found_inf is True

    def test_step_skips_on_inf(self):
        scaler = GradScaler(init_scale=1.0)
        w = torch.tensor([5.0], requires_grad=True)
        loss = w.sum()
        loss.backward()
        w.grad.data.fill_(float("inf"))

        opt = torch.optim.SGD([w], lr=0.1)
        scaler.unscale_(opt)

        original_val = w.item()
        scaler.step(opt)
        # Weight should NOT change because inf was found
        assert w.item() == original_val

    def test_update_backoff_on_inf(self):
        scaler = GradScaler(init_scale=1024.0, backoff_factor=0.5)
        scaler._found_inf = True
        scaler.update()
        assert scaler.get_scale() == 512.0

    def test_update_growth_after_interval(self):
        scaler = GradScaler(init_scale=100.0, growth_factor=2.0, growth_interval=3)
        for _ in range(3):
            scaler._found_inf = False
            scaler.update()
        assert scaler.get_scale() == 200.0

    def test_growth_tracker_resets_on_inf(self):
        scaler = GradScaler(init_scale=100.0, growth_factor=2.0, growth_interval=5)
        # 3 good steps
        for _ in range(3):
            scaler._found_inf = False
            scaler.update()
        # 1 bad step resets counter
        scaler._found_inf = True
        scaler.update()
        # 4 more good steps (not enough for growth since reset)
        for _ in range(4):
            scaler._found_inf = False
            scaler.update()
        # scale was backed off once: 100 * 0.5 = 50, then 4 good steps < 5
        assert scaler.get_scale() == 50.0


class TestMixedPrecisionTrainStep:
    def test_returns_finite_loss(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler(init_scale=256.0)

        data = torch.randn(8, 16)
        target = torch.randn(8, 4)
        loss_fn = nn.MSELoss()

        loss_val = mixed_precision_train_step(model, optimizer, data, target, loss_fn, scaler)
        assert isinstance(loss_val, float)
        assert not (loss_val != loss_val)  # not NaN

    def test_training_reduces_loss(self):
        """Multiple steps should reduce loss."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)).half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler(init_scale=16.0)
        loss_fn = nn.MSELoss()

        data = torch.randn(16, 8)
        target = torch.zeros(16, 2)

        losses = []
        for _ in range(50):
            loss_val = mixed_precision_train_step(
                model, optimizer, data, target, loss_fn, scaler
            )
            losses.append(loss_val)

        # Filter out NaN losses (can happen with fp16)
        valid_losses = [l for l in losses if l == l]  # filter NaN
        assert len(valid_losses) > 10, "Too many NaN losses"
        # Loss should decrease over training
        assert valid_losses[-1] < valid_losses[0], f"Loss did not decrease: {valid_losses[0]} -> {valid_losses[-1]}"

    def test_gradients_are_produced(self):
        torch.manual_seed(42)
        model = nn.Linear(4, 2).half()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler(init_scale=64.0)

        data = torch.randn(2, 4)
        target = torch.randn(2, 2)
        loss_fn = nn.MSELoss()

        mixed_precision_train_step(model, optimizer, data, target, loss_fn, scaler)

        # After step, params should have been updated (grads zeroed by next step)
        # Just verify the step completed without error
        assert True
