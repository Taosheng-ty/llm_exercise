import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

split_qkv_weight = _mod.split_qkv_weight
fuse_qkv_weight = _mod.fuse_qkv_weight


class TestSplitQKV:
    """Test QKV splitting for various head configurations."""

    def test_mha_basic(self):
        """MHA: num_q_heads == num_kv_heads (no GQA)."""
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 8, 8, 64, 512
        total = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused = torch.randn(total, hidden_dim)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        assert q.shape == (num_q_heads * head_dim, hidden_dim)
        assert k.shape == (num_kv_heads * head_dim, hidden_dim)
        assert v.shape == (num_kv_heads * head_dim, hidden_dim)

    def test_gqa_4_to_1(self):
        """GQA: 4 query heads per KV head."""
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 32, 8, 128, 4096
        total = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused = torch.randn(total, hidden_dim)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        assert q.shape == (32 * 128, 4096)
        assert k.shape == (8 * 128, 4096)
        assert v.shape == (8 * 128, 4096)

    def test_roundtrip_mha(self):
        """Roundtrip: fuse(split(x)) == x for MHA."""
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 4, 4, 32, 128
        total = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused = torch.randn(total, hidden_dim)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)
        reconstructed = fuse_qkv_weight(q, k, v, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        assert torch.allclose(fused, reconstructed)

    def test_roundtrip_gqa(self):
        """Roundtrip: fuse(split(x)) == x for GQA."""
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 16, 4, 64, 1024
        total = (num_q_heads + 2 * num_kv_heads) * head_dim
        fused = torch.randn(total, hidden_dim)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)
        reconstructed = fuse_qkv_weight(q, k, v, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        assert torch.allclose(fused, reconstructed)

    def test_values_correct(self):
        """Verify actual values in a small deterministic case."""
        # 2 q heads, 1 kv head, head_dim=2, hidden=3
        # Layout: [q0, q1, k0, v0] where each is (head_dim, hidden) = (2, 3)
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 2, 1, 2, 3
        fused = torch.arange(24, dtype=torch.float32).reshape(8, 3)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        # q should be rows 0-3 (2 heads * 2 dim)
        assert torch.equal(q, fused[0:4])
        # k should be rows 4-5
        assert torch.equal(k, fused[4:6])
        # v should be rows 6-7
        assert torch.equal(v, fused[6:8])

    def test_gqa_2groups_values(self):
        """GQA with 2 kv groups, verify interleaved layout is handled."""
        num_q_heads, num_kv_heads, head_dim, hidden_dim = 4, 2, 2, 3
        # Each group: 2 Q heads + 1 K + 1 V = 4 rows of head_dim=2 = 8 rows per group
        # Total: 2 groups * 8 rows = 16 rows
        total = (num_q_heads + 2 * num_kv_heads) * head_dim
        assert total == 16
        fused = torch.arange(48, dtype=torch.float32).reshape(16, 3)

        q, k, v = split_qkv_weight(fused, num_q_heads, num_kv_heads, head_dim, hidden_dim)

        # Group 0: rows 0-7 -> Q(0:4), K(4:6), V(6:8)
        # Group 1: rows 8-15 -> Q(8:12), K(12:14), V(14:16)
        expected_q = torch.cat([fused[0:4], fused[8:12]], dim=0)
        expected_k = torch.cat([fused[4:6], fused[12:14]], dim=0)
        expected_v = torch.cat([fused[6:8], fused[14:16]], dim=0)

        assert torch.equal(q, expected_q)
        assert torch.equal(k, expected_k)
        assert torch.equal(v, expected_v)
