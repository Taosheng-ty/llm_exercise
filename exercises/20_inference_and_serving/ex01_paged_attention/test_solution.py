"""
Tests for Exercise 01: Paged Attention KV Cache
"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PagedKVCache = _mod.PagedKVCache


class TestPagedKVCacheInit:
    def test_initial_free_blocks(self):
        cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=16)
        assert cache.num_free_blocks() == 8

    def test_cache_storage_shape(self):
        cache = PagedKVCache(num_blocks=4, block_size=3, num_heads=2, head_dim=8)
        assert cache.key_cache.shape == (4, 3, 2, 8)
        assert cache.value_cache.shape == (4, 3, 2, 8)


class TestAllocateAndFree:
    def test_allocate_reduces_free_count(self):
        cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=16)
        blocks = cache.allocate_blocks(3)
        assert len(blocks) == 3
        assert cache.num_free_blocks() == 5

    def test_allocate_returns_unique_indices(self):
        cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=16)
        blocks = cache.allocate_blocks(8)
        assert len(set(blocks)) == 8

    def test_allocate_too_many_raises(self):
        cache = PagedKVCache(num_blocks=4, block_size=4, num_heads=2, head_dim=16)
        with pytest.raises(RuntimeError):
            cache.allocate_blocks(5)

    def test_free_blocks_restores_count(self):
        cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=16)
        blocks = cache.allocate_blocks(4)
        cache.free_blocks(blocks[:2])
        assert cache.num_free_blocks() == 6


class TestAppendAndRead:
    def test_append_single_token(self):
        cache = PagedKVCache(num_blocks=4, block_size=4, num_heads=2, head_dim=8)
        key = torch.randn(2, 8)
        value = torch.randn(2, 8)
        block_table = cache.append_token([], key, value)
        assert len(block_table) == 1
        assert cache.num_free_blocks() == 3

        keys, values = cache.read_kv(block_table, seq_len=1)
        assert keys.shape == (1, 2, 8)
        assert torch.allclose(keys[0], key)
        assert torch.allclose(values[0], value)

    def test_append_fills_block_then_allocates_new(self):
        """Appending more tokens than block_size should span multiple blocks."""
        cache = PagedKVCache(num_blocks=8, block_size=2, num_heads=1, head_dim=4)
        block_table = []
        tokens_k = []
        tokens_v = []
        for i in range(5):
            k = torch.full((1, 4), float(i))
            v = torch.full((1, 4), float(i + 100))
            block_table = cache.append_token(block_table, k, v)
            tokens_k.append(k)
            tokens_v.append(v)

        # 5 tokens with block_size=2 needs 3 blocks (2+2+1)
        assert len(block_table) == 3
        assert cache.num_free_blocks() == 5

        keys, values = cache.read_kv(block_table, seq_len=5)
        assert keys.shape == (5, 1, 4)
        for i in range(5):
            assert torch.allclose(keys[i], tokens_k[i])
            assert torch.allclose(values[i], tokens_v[i])

    def test_read_kv_correct_across_boundaries(self):
        """Read-back should be correct even when data crosses block boundaries."""
        cache = PagedKVCache(num_blocks=10, block_size=3, num_heads=2, head_dim=4)
        block_table = []
        seq_len = 7
        expected_keys = torch.randn(seq_len, 2, 4)
        expected_vals = torch.randn(seq_len, 2, 4)
        for t in range(seq_len):
            block_table = cache.append_token(block_table, expected_keys[t], expected_vals[t])

        # 7 tokens, block_size=3 -> 3 blocks (3+3+1)
        assert len(block_table) == 3

        keys, values = cache.read_kv(block_table, seq_len)
        assert torch.allclose(keys, expected_keys)
        assert torch.allclose(values, expected_vals)

    def test_multiple_sequences_independent(self):
        """Two sequences should use independent blocks."""
        cache = PagedKVCache(num_blocks=10, block_size=2, num_heads=1, head_dim=4)

        bt1, bt2 = [], []
        for i in range(3):
            k1 = torch.full((1, 4), 1.0)
            v1 = torch.full((1, 4), 10.0)
            bt1 = cache.append_token(bt1, k1, v1)

            k2 = torch.full((1, 4), 2.0)
            v2 = torch.full((1, 4), 20.0)
            bt2 = cache.append_token(bt2, k2, v2)

        k1_read, v1_read = cache.read_kv(bt1, 3)
        k2_read, v2_read = cache.read_kv(bt2, 3)

        assert torch.allclose(k1_read, torch.full((3, 1, 4), 1.0))
        assert torch.allclose(v1_read, torch.full((3, 1, 4), 10.0))
        assert torch.allclose(k2_read, torch.full((3, 1, 4), 2.0))
        assert torch.allclose(v2_read, torch.full((3, 1, 4), 20.0))

        # No overlapping blocks
        assert set(bt1).isdisjoint(set(bt2))

    def test_free_and_reuse(self):
        """After freeing blocks, they can be reallocated."""
        cache = PagedKVCache(num_blocks=4, block_size=2, num_heads=1, head_dim=4)
        bt = []
        for _ in range(4):
            bt = cache.append_token(bt, torch.zeros(1, 4), torch.zeros(1, 4))
        assert cache.num_free_blocks() == 2

        cache.free_blocks(bt)
        assert cache.num_free_blocks() == 4

        # Re-allocate should work
        bt2 = []
        for _ in range(4):
            bt2 = cache.append_token(bt2, torch.ones(1, 4), torch.ones(1, 4))
        keys, _ = cache.read_kv(bt2, 4)
        assert torch.allclose(keys, torch.ones(4, 1, 4))
