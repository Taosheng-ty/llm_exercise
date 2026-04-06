import importlib.util
import os
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MultiHeadAttention = _mod.MultiHeadAttention


def test_output_shape():
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out = mha(x)
    assert out.shape == (2, 8, 64)


def test_causal_mode_runs():
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out = mha(x, causal=True)
    assert out.shape == (2, 8, 64)


def test_different_num_heads():
    for nh in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model=64, num_heads=nh)
        x = torch.randn(1, 4, 64)
        out = mha(x)
        assert out.shape == (1, 4, 64)


def test_deterministic():
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model=32, num_heads=2)
    x = torch.randn(1, 4, 32)
    out1 = mha(x)
    out2 = mha(x)
    assert torch.allclose(out1, out2)


def test_gradient_flows():
    mha = MultiHeadAttention(d_model=32, num_heads=2)
    x = torch.randn(1, 4, 32, requires_grad=True)
    out = mha(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
