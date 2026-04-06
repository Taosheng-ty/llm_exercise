"""Tests for Exercise 03: ZeRO Optimizer Sharding"""

import importlib.util
import os

import numpy as np
import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
shard_optimizer_state = _mod.shard_optimizer_state
gather_for_step = _mod.gather_for_step
reduce_scatter_gradients = _mod.reduce_scatter_gradients
shard_parameters = _mod.shard_parameters
all_gather_params = _mod.all_gather_params


class TestShardOptimizerState:
    def test_correct_number_of_shards(self):
        """Should return one state dict per rank."""
        state = {"m": np.arange(8, dtype=np.float32), "v": np.arange(8, dtype=np.float32)}
        shards = shard_optimizer_state(state, num_ranks=4)
        assert len(shards) == 4

    def test_shard_sizes_even(self):
        """Even split: each shard should have num_params/num_ranks elements."""
        state = {"m": np.arange(12, dtype=np.float32), "v": np.ones(12, dtype=np.float32)}
        shards = shard_optimizer_state(state, num_ranks=3)
        for s in shards:
            assert len(s["m"]) == 4
            assert len(s["v"]) == 4

    def test_shard_sizes_uneven(self):
        """Uneven split: last rank should get the remainder."""
        state = {"m": np.arange(10, dtype=np.float32), "v": np.ones(10, dtype=np.float32)}
        shards = shard_optimizer_state(state, num_ranks=3)
        assert len(shards[0]["m"]) == 3
        assert len(shards[1]["m"]) == 3
        assert len(shards[2]["m"]) == 4  # remainder

    def test_shard_content_correct(self):
        """Shard contents should match corresponding slices of the original."""
        m = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        shards = shard_optimizer_state({"m": m, "v": v}, num_ranks=2)
        np.testing.assert_array_equal(shards[0]["m"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(shards[1]["v"], [0.4, 0.5, 0.6])


class TestGatherForStep:
    def test_roundtrip(self):
        """Shard then gather should reconstruct the original state."""
        state = {"m": np.random.randn(20).astype(np.float32),
                 "v": np.random.randn(20).astype(np.float32)}
        shards = shard_optimizer_state(state, num_ranks=4)
        gathered = gather_for_step(shards, rank=0)
        np.testing.assert_array_almost_equal(gathered["m"], state["m"])
        np.testing.assert_array_almost_equal(gathered["v"], state["v"])

    def test_roundtrip_uneven(self):
        """Shard then gather should work for uneven splits."""
        state = {"m": np.arange(11, dtype=np.float32),
                 "v": np.arange(11, dtype=np.float32) * 2}
        shards = shard_optimizer_state(state, num_ranks=3)
        gathered = gather_for_step(shards, rank=2)
        np.testing.assert_array_almost_equal(gathered["m"], state["m"])
        np.testing.assert_array_almost_equal(gathered["v"], state["v"])


class TestReduceScatterGradients:
    def test_correct_number_of_shards(self):
        """Should return one gradient shard per rank."""
        grads = np.arange(12, dtype=np.float32)
        shards = reduce_scatter_gradients(grads, num_ranks=3)
        assert len(shards) == 3

    def test_shards_cover_full_gradient(self):
        """Concatenating shards should reconstruct the full gradient."""
        grads = np.random.randn(15).astype(np.float32)
        shards = reduce_scatter_gradients(grads, num_ranks=4)
        reconstructed = np.concatenate(shards)
        np.testing.assert_array_almost_equal(reconstructed, grads)


class TestShardAndGatherParams:
    def test_shard_correct_count(self):
        """Should return one shard per rank."""
        params = np.arange(16, dtype=np.float32)
        shards = shard_parameters(params, num_ranks=4)
        assert len(shards) == 4

    def test_all_gather_roundtrip(self):
        """Shard then all-gather should reconstruct original parameters."""
        params = np.random.randn(20).astype(np.float32)
        shards = shard_parameters(params, num_ranks=5)
        gathered = all_gather_params(shards)
        np.testing.assert_array_almost_equal(gathered, params)

    def test_all_gather_uneven(self):
        """Uneven parameter split should still round-trip correctly."""
        params = np.arange(13, dtype=np.float32)
        shards = shard_parameters(params, num_ranks=3)
        gathered = all_gather_params(shards)
        np.testing.assert_array_almost_equal(gathered, params)
