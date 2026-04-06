import importlib.util
import os
import torch
import torch.nn.functional as F
import math

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

KVCache = _mod.KVCache
incremental_attention = _mod.incremental_attention


def test_cache_accumulates():
    cache = KVCache(max_seq_len=100)
    k1 = torch.randn(1, 2, 3, 8)  # 3 tokens
    v1 = torch.randn(1, 2, 3, 8)
    full_k, full_v = cache.update(k1, v1)
    assert full_k.shape == (1, 2, 3, 8)

    k2 = torch.randn(1, 2, 2, 8)  # 2 more tokens
    v2 = torch.randn(1, 2, 2, 8)
    full_k, full_v = cache.update(k2, v2)
    assert full_k.shape == (1, 2, 5, 8)


def test_cache_eviction():
    cache = KVCache(max_seq_len=4)
    for i in range(6):
        k = torch.randn(1, 1, 1, 8)
        v = torch.randn(1, 1, 1, 8)
        full_k, full_v = cache.update(k, v)
    assert full_k.size(2) == 4  # max_seq_len


def test_cache_reset():
    cache = KVCache(max_seq_len=100)
    cache.update(torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8))
    cache.reset()
    assert cache.seq_len == 0


def test_incremental_matches_full():
    """Incremental decoding should produce the same result as full attention."""
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 6, 8
    Q_full = torch.randn(B, H, S, D)
    K_full = torch.randn(B, H, S, D)
    V_full = torch.randn(B, H, S, D)

    # Full attention for the last token
    scores = torch.matmul(Q_full[:, :, -1:, :], K_full.transpose(-2, -1)) / math.sqrt(D)
    attn = F.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V_full)

    # Incremental: feed tokens one by one
    cache = KVCache(max_seq_len=100)
    for i in range(S - 1):
        cache.update(K_full[:, :, i:i+1, :], V_full[:, :, i:i+1, :])
    out = incremental_attention(
        Q_full[:, :, -1:, :],
        K_full[:, :, -1:, :],
        V_full[:, :, -1:, :],
        cache,
    )
    assert torch.allclose(out, ref, atol=1e-5)


def test_incremental_step_by_step():
    """Each incremental step should produce valid output."""
    cache = KVCache(max_seq_len=100)
    B, H, D = 1, 2, 8
    for step in range(5):
        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        out = incremental_attention(q, k, v, cache)
        assert out.shape == (B, H, 1, D)
        assert not torch.isnan(out).any()
    assert cache.seq_len == 5
