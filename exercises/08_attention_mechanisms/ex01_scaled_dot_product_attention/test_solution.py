import importlib.util
import os
import torch
import torch.nn.functional as F
import math

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

scaled_dot_product_attention = _mod.scaled_dot_product_attention


def test_output_shape():
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out, attn = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, H, S, D)
    assert attn.shape == (B, H, S, S)


def test_attention_weights_sum_to_one():
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    _, attn = scaled_dot_product_attention(Q, K, V)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_matches_pytorch_no_causal():
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out, _ = scaled_dot_product_attention(Q, K, V, causal=False)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    assert torch.allclose(out, ref, atol=1e-5)


def test_matches_pytorch_causal():
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out, _ = scaled_dot_product_attention(Q, K, V, causal=True)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_causal_mask_zeros_future():
    B, H, S, D = 1, 1, 4, 8
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    _, attn = scaled_dot_product_attention(Q, K, V, causal=True)
    # Upper triangle (future) should be zero
    for i in range(S):
        for j in range(i + 1, S):
            assert attn[0, 0, i, j].item() == 0.0
