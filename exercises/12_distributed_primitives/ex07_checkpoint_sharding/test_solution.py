"""
Tests for Exercise 07: Checkpoint Sharding
"""

import importlib.util
import os

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

shard_state_dict = _mod.shard_state_dict
reconstruct_state_dict = _mod.reconstruct_state_dict


def _make_state_dict():
    np.random.seed(42)
    return {
        "layer0.weight": np.random.randn(64, 32).astype(np.float32),
        "layer0.bias": np.random.randn(64).astype(np.float32),
        "layer1.weight": np.random.randn(32, 64).astype(np.float32),
        "layer1.bias": np.random.randn(32).astype(np.float32),
        "layer2.weight": np.random.randn(16, 32).astype(np.float32),
        "layer2.bias": np.random.randn(16).astype(np.float32),
    }


def test_shard_count():
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 3)
    assert len(shards) == 3


def test_all_tensors_in_index():
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 2)
    assert set(index.keys()) == set(sd.keys())


def test_index_format():
    sd = _make_state_dict()
    _, index = shard_state_dict(sd, 3)
    for name, shard_file in index.items():
        assert shard_file.startswith("shard_")
        assert shard_file.endswith(".bin")
        # Extract index number
        idx_str = shard_file[len("shard_"):-len(".bin")]
        idx = int(idx_str)
        assert 0 <= idx < 3


def test_roundtrip_reconstruction():
    """Reconstructed state dict must match original exactly."""
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 3)
    reconstructed = reconstruct_state_dict(shards, index)
    assert set(reconstructed.keys()) == set(sd.keys())
    for name in sd:
        np.testing.assert_array_equal(reconstructed[name], sd[name])


def test_balanced_sharding():
    """Shard sizes should be roughly balanced."""
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 2)
    sizes = [sum(v.size for v in shard.values()) for shard in shards]
    total = sum(sizes)
    # Each shard should have between 30% and 70% of total
    for s in sizes:
        assert s >= 0.3 * total, f"Shard too small: {s} vs total {total}"
        assert s <= 0.7 * total, f"Shard too large: {s} vs total {total}"


def test_no_tensor_split_across_shards():
    """Each tensor should appear in exactly one shard."""
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 3)
    all_names = []
    for shard in shards:
        all_names.extend(shard.keys())
    assert len(all_names) == len(sd), "Some tensors are duplicated or missing"
    assert set(all_names) == set(sd.keys())


def test_single_shard():
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 1)
    assert len(shards) == 1
    assert set(shards[0].keys()) == set(sd.keys())
    reconstructed = reconstruct_state_dict(shards, index)
    for name in sd:
        np.testing.assert_array_equal(reconstructed[name], sd[name])


def test_many_shards():
    """More shards than tensors: some shards may be empty."""
    sd = _make_state_dict()
    shards, index = shard_state_dict(sd, 10)
    assert len(shards) == 10
    reconstructed = reconstruct_state_dict(shards, index)
    for name in sd:
        np.testing.assert_array_equal(reconstructed[name], sd[name])


def test_scalar_tensors():
    sd = {"a": np.array(1.0), "b": np.array(2.0), "c": np.array(3.0)}
    shards, index = shard_state_dict(sd, 2)
    reconstructed = reconstruct_state_dict(shards, index)
    for name in sd:
        np.testing.assert_array_equal(reconstructed[name], sd[name])
