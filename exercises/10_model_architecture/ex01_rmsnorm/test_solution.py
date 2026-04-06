"""Tests for Exercise 01: RMSNorm"""

import importlib.util
import os
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RMSNorm = _mod.RMSNorm


def test_output_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"


def test_weight_is_parameter():
    norm = RMSNorm(32)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.shape == (32,)
    assert torch.allclose(norm.weight.data, torch.ones(32))


def test_normalization_unit_rms():
    """After RMSNorm with weight=1, the RMS of each vector should be ~1."""
    norm = RMSNorm(128)
    x = torch.randn(4, 8, 128) * 5.0 + 3.0  # shifted and scaled
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
        f"RMS should be ~1 after normalization, got mean={rms.mean().item():.4f}"
    )


def test_no_mean_subtraction():
    """RMSNorm should NOT subtract the mean (unlike LayerNorm)."""
    norm = RMSNorm(64)
    # Use a constant input -- mean is nonzero
    x = torch.ones(2, 4, 64) * 3.0
    out = norm(x)
    # If mean were subtracted, output would be 0. With RMSNorm it should be ~1.0
    assert out.abs().mean() > 0.5, "RMSNorm should not subtract mean"


def test_learnable_weight():
    """Changing weight should change the output."""
    norm = RMSNorm(32)
    x = torch.randn(1, 5, 32)
    out1 = norm(x).clone()
    norm.weight.data.fill_(2.0)
    out2 = norm(x)
    assert torch.allclose(out2, out1 * 2.0, atol=1e-5), "Weight should scale output"


def test_gradient_flows():
    norm = RMSNorm(16)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
