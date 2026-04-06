"""Tests for Exercise 07: Residual Connections with Scaling"""

import importlib.util
import os
import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PreNormResidual = _mod.PreNormResidual


def test_identity_sublayer():
    """With identity sublayer, output = x * alpha + norm(x)."""
    identity = nn.Identity()
    block = PreNormResidual(dim=16, sublayer=identity, alpha=1.0)
    x = torch.randn(2, 4, 16)
    out = block(x)
    assert out.shape == x.shape


def test_residual_connection():
    """Output should contain the original input (residual)."""
    # Use a zero sublayer: sublayer always returns 0
    zero_layer = nn.Linear(16, 16, bias=False)
    nn.init.zeros_(zero_layer.weight)
    block = PreNormResidual(dim=16, sublayer=zero_layer, alpha=1.0)
    x = torch.randn(2, 4, 16)
    out = block(x)
    # sublayer output is 0, so output should be x * 1.0 + 0 = x
    assert torch.allclose(out, x, atol=1e-5), "With zero sublayer, output should equal input"


def test_alpha_scaling():
    """Alpha should scale the residual."""
    zero_layer = nn.Linear(16, 16, bias=False)
    nn.init.zeros_(zero_layer.weight)
    block = PreNormResidual(dim=16, sublayer=zero_layer, alpha=0.5)
    x = torch.randn(2, 4, 16)
    out = block(x)
    assert torch.allclose(out, x * 0.5, atol=1e-5), "With zero sublayer, output = x * alpha"


def test_gradient_flows_through_residual():
    """Gradients should flow through both the residual and sublayer paths."""
    linear = nn.Linear(16, 16, bias=False)
    block = PreNormResidual(dim=16, sublayer=linear, alpha=1.0)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient should flow to input"
    # Gradient should be nonzero (not vanished)
    assert x.grad.abs().mean() > 1e-6, "Gradient should be non-trivial"


def test_pre_norm_pattern():
    """Sublayer should receive normalized input, not raw input."""
    class RecordInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x.clone()
            return x

    recorder = RecordInput()
    block = PreNormResidual(dim=16, sublayer=recorder, alpha=1.0)
    x = torch.randn(2, 4, 16) * 10.0  # large values
    block(x)

    # The recorded input should be normalized (RMS ~1), not raw
    rms = recorder.last_input.pow(2).mean(dim=-1).sqrt()
    assert rms.mean() < 5.0, "Sublayer should receive normalized input"
