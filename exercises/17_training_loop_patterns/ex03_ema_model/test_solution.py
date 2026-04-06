"""Tests for Exercise 03: Exponential Moving Average (EMA) of Model Weights"""

import importlib.util
import os
import torch
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EMAModel = _mod.EMAModel


def _make_model():
    torch.manual_seed(42)
    return torch.nn.Linear(4, 2, bias=True)


class TestEMAModel:
    def test_initial_ema_equals_model(self):
        """EMA should start as a copy of model parameters."""
        model = _make_model()
        ema = EMAModel(model, decay=0.999)
        for ema_p, model_p in zip(ema.shadow_params, model.parameters()):
            torch.testing.assert_close(ema_p, model_p.data)

    def test_ema_is_independent_copy(self):
        """Changing model should not change EMA (before update)."""
        model = _make_model()
        ema = EMAModel(model, decay=0.999)
        orig_ema = [p.clone() for p in ema.shadow_params]
        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        # EMA should be unchanged
        for ema_p, orig_p in zip(ema.shadow_params, orig_ema):
            torch.testing.assert_close(ema_p, orig_p)

    def test_single_update(self):
        """After one update, EMA = decay * old + (1-decay) * new."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9)
        old_ema = [p.clone() for p in ema.shadow_params]
        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(10.0)
        ema.update(model)
        for ema_p, old_p in zip(ema.shadow_params, old_ema):
            expected = 0.9 * old_p + 0.1 * 10.0
            torch.testing.assert_close(ema_p, expected)

    def test_copy_to(self):
        """copy_to should overwrite model weights with EMA weights."""
        model = _make_model()
        ema = EMAModel(model, decay=0.5)
        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(100.0)
        ema.update(model)
        # Now copy EMA back
        ema.copy_to(model)
        for ema_p, model_p in zip(ema.shadow_params, model.parameters()):
            torch.testing.assert_close(model_p.data, ema_p)

    def test_ema_smooths_noisy_updates(self):
        """EMA with high decay should be smoother than raw params."""
        model = _make_model()
        ema = EMAModel(model, decay=0.99)
        torch.manual_seed(123)
        deltas = []
        ema_deltas = []
        prev_ema = ema.shadow_params[0].clone()
        prev_model = list(model.parameters())[0].data.clone()

        for _ in range(50):
            noise = torch.randn_like(list(model.parameters())[0]) * 5.0
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(noise[:p.shape[0], :p.shape[1]] if p.dim() == 2 else noise[:p.shape[0], 0])
            ema.update(model)
            curr_model = list(model.parameters())[0].data.clone()
            curr_ema = ema.shadow_params[0].clone()
            deltas.append((curr_model - prev_model).norm().item())
            ema_deltas.append((curr_ema - prev_ema).norm().item())
            prev_model = curr_model
            prev_ema = curr_ema

        # EMA changes should be smaller on average (smoother)
        avg_model_delta = sum(deltas) / len(deltas)
        avg_ema_delta = sum(ema_deltas) / len(ema_deltas)
        assert avg_ema_delta < avg_model_delta

    def test_state_dict_roundtrip(self):
        """state_dict / load_state_dict should preserve EMA state."""
        model = _make_model()
        ema = EMAModel(model, decay=0.95)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(5.0)
        ema.update(model)

        sd = ema.state_dict()
        ema2 = EMAModel(model, decay=0.5)  # different decay
        ema2.load_state_dict(sd)

        assert ema2.decay == 0.95
        for p1, p2 in zip(ema.shadow_params, ema2.shadow_params):
            torch.testing.assert_close(p1, p2)

    def test_decay_zero_means_no_memory(self):
        """With decay=0, EMA should equal the latest model params."""
        model = _make_model()
        ema = EMAModel(model, decay=0.0)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(42.0)
        ema.update(model)
        for ema_p in ema.shadow_params:
            assert torch.allclose(ema_p, torch.tensor(42.0))

    def test_decay_one_means_no_update(self):
        """With decay=1, EMA should never change from initialization."""
        model = _make_model()
        ema = EMAModel(model, decay=1.0)
        orig = [p.clone() for p in ema.shadow_params]
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        ema.update(model)
        for ema_p, orig_p in zip(ema.shadow_params, orig):
            torch.testing.assert_close(ema_p, orig_p)
