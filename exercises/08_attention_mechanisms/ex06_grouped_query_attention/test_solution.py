import importlib.util
import os
import torch
import torch.nn.functional as F
import math

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

grouped_query_attention = _mod.grouped_query_attention


def test_output_shape():
    B, S, D = 2, 8, 64
    num_q_heads, num_kv_heads = 8, 2
    head_dim = D // num_q_heads
    Q = torch.randn(B, S, num_q_heads * head_dim)
    K = torch.randn(B, S, num_kv_heads * head_dim)
    V = torch.randn(B, S, num_kv_heads * head_dim)
    out = grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads)
    assert out.shape == (B, S, num_q_heads * head_dim)


def test_mha_case():
    """When num_kv_heads == num_q_heads, GQA reduces to standard MHA."""
    torch.manual_seed(0)
    B, S, num_heads, head_dim = 1, 4, 4, 8
    D = num_heads * head_dim
    Q = torch.randn(B, S, D)
    K = torch.randn(B, S, D)
    V = torch.randn(B, S, D)
    out = grouped_query_attention(Q, K, V, num_heads, num_heads)
    assert out.shape == (B, S, D)

    # Compare with direct attention
    q = Q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = K.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = V.view(B, S, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    ref = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
    assert torch.allclose(out, ref, atol=1e-5)


def test_mqa_case():
    """When num_kv_heads == 1, it is Multi-Query Attention."""
    B, S, num_q_heads, head_dim = 2, 8, 8, 8
    Q = torch.randn(B, S, num_q_heads * head_dim)
    K = torch.randn(B, S, 1 * head_dim)
    V = torch.randn(B, S, 1 * head_dim)
    out = grouped_query_attention(Q, K, V, num_q_heads, 1)
    assert out.shape == (B, S, num_q_heads * head_dim)


def test_causal():
    B, S, num_q_heads, num_kv_heads, head_dim = 1, 4, 4, 2, 8
    Q = torch.randn(B, S, num_q_heads * head_dim)
    K = torch.randn(B, S, num_kv_heads * head_dim)
    V = torch.randn(B, S, num_kv_heads * head_dim)
    out = grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads, causal=True)
    assert out.shape == (B, S, num_q_heads * head_dim)


def test_kv_sharing():
    """All Q heads in a group should use the same K,V head."""
    torch.manual_seed(1)
    B, S, head_dim = 1, 4, 8
    num_q_heads, num_kv_heads = 4, 1
    Q = torch.randn(B, S, num_q_heads * head_dim)
    K = torch.randn(B, S, num_kv_heads * head_dim)
    V = torch.randn(B, S, num_kv_heads * head_dim)

    out = grouped_query_attention(Q, K, V, num_q_heads, num_kv_heads)
    # Reshape to per-head outputs
    out_heads = out.view(B, S, num_q_heads, head_dim)
    # All heads share same K,V so different Q heads produce different results
    # but the function should still run and produce valid output
    assert not torch.isnan(out_heads).any()
