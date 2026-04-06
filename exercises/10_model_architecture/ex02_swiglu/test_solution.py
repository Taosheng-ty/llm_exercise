"""Tests for Exercise 02: SwiGLU Activation"""

import importlib.util
import os
import torch
import torch.nn.functional as F

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SwiGLUFFN = _mod.SwiGLUFFN


def test_output_shape():
    ffn = SwiGLUFFN(dim=64, hidden_dim=128)
    x = torch.randn(2, 10, 64)
    out = ffn(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"


def test_no_bias():
    ffn = SwiGLUFFN(dim=32, hidden_dim=64)
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_three_projections():
    ffn = SwiGLUFFN(dim=32, hidden_dim=64)
    assert ffn.gate_proj.in_features == 32
    assert ffn.gate_proj.out_features == 64
    assert ffn.up_proj.in_features == 32
    assert ffn.up_proj.out_features == 64
    assert ffn.down_proj.in_features == 64
    assert ffn.down_proj.out_features == 32


def test_silu_gating_behavior():
    """Verify the gating mechanism uses silu."""
    torch.manual_seed(42)
    ffn = SwiGLUFFN(dim=16, hidden_dim=32)
    x = torch.randn(1, 4, 16)

    # Manual computation
    gate = F.silu(ffn.gate_proj(x))
    up = ffn.up_proj(x)
    expected = ffn.down_proj(gate * up)

    out = ffn(x)
    assert torch.allclose(out, expected, atol=1e-6), "Output should match manual SwiGLU computation"


def test_gradient_flows():
    ffn = SwiGLUFFN(dim=16, hidden_dim=32)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = ffn(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_different_hidden_dim():
    """hidden_dim can be different from dim."""
    ffn = SwiGLUFFN(dim=64, hidden_dim=256)
    x = torch.randn(1, 5, 64)
    out = ffn(x)
    assert out.shape == (1, 5, 64)
