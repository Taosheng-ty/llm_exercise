import importlib.util
import os
import torch
import torch.nn.functional as F
import math

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

flash_attention_tiling = _mod.flash_attention_tiling


def _naive_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def test_matches_naive_small():
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = flash_attention_tiling(Q, K, V, block_size=4)
    ref = _naive_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5)


def test_matches_naive_larger():
    torch.manual_seed(1)
    B, H, S, D = 2, 4, 32, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = flash_attention_tiling(Q, K, V, block_size=8)
    ref = _naive_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5)


def test_different_block_sizes():
    torch.manual_seed(2)
    B, H, S, D = 1, 2, 16, 8
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    ref = _naive_attention(Q, K, V)
    for bs in [1, 4, 7, 16]:
        out = flash_attention_tiling(Q, K, V, block_size=bs)
        assert torch.allclose(out, ref, atol=1e-4), f"Failed for block_size={bs}"


def test_output_shape():
    B, H, S, D = 2, 4, 16, 32
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = flash_attention_tiling(Q, K, V, block_size=4)
    assert out.shape == (B, H, S, D)


def test_block_size_larger_than_seq():
    torch.manual_seed(3)
    B, H, S, D = 1, 1, 4, 8
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = flash_attention_tiling(Q, K, V, block_size=64)
    ref = _naive_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5)
