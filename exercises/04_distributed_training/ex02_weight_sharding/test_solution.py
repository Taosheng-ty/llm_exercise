"""Tests for Exercise 02: Weight Sharding."""

import importlib.util
import os
import numpy as np
import pytest

# Load solution from same directory
_spec = importlib.util.spec_from_file_location(
    "solution_ex02", os.path.join(os.path.dirname(__file__), "solution.py")
)
_sol = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sol)
shard_weights = _sol.shard_weights
gather_weights = _sol.gather_weights


def test_even_split():
    """Tensors that divide evenly by num_shards."""
    weights = {
        "layer1.weight": np.arange(12).reshape(4, 3).astype(np.float32),
        "layer1.bias": np.arange(4).astype(np.float32),
    }
    shards, metadata = shard_weights(weights, num_shards=2)

    assert len(shards) == 2
    assert shards[0]["layer1.weight"].shape == (2, 3)
    assert shards[1]["layer1.weight"].shape == (2, 3)
    assert shards[0]["layer1.bias"].shape == (2,)
    np.testing.assert_array_equal(
        shards[0]["layer1.weight"], np.array([[0, 1, 2], [3, 4, 5]])
    )


def test_uneven_split_with_padding():
    """Tensors that don't divide evenly should be padded then split."""
    weights = {"w": np.arange(15).reshape(5, 3).astype(np.float32)}
    shards, metadata = shard_weights(weights, num_shards=3)

    # 5 rows -> pad to 6 -> 3 shards of 2 rows each
    assert len(shards) == 3
    for s in shards:
        assert s["w"].shape == (2, 3)

    # Last shard's second row should be zeros (padding)
    np.testing.assert_array_equal(shards[2]["w"][1], [0, 0, 0])


def test_roundtrip_even():
    """Shard then gather should recover original weights exactly."""
    weights = {
        "attn.qkv": np.random.randn(8, 16).astype(np.float32),
        "ffn.up": np.random.randn(16, 8).astype(np.float32),
    }
    shards, metadata = shard_weights(weights, num_shards=4)
    recovered = gather_weights(shards, metadata)

    for name in weights:
        np.testing.assert_array_equal(recovered[name], weights[name])


def test_roundtrip_uneven():
    """Shard+gather roundtrip with padding should recover originals."""
    weights = {
        "embed": np.random.randn(7, 32).astype(np.float32),
        "head": np.random.randn(10, 5).astype(np.float32),
    }
    shards, metadata = shard_weights(weights, num_shards=3)
    recovered = gather_weights(shards, metadata)

    for name in weights:
        assert recovered[name].shape == weights[name].shape
        np.testing.assert_array_almost_equal(recovered[name], weights[name])


def test_single_shard():
    """With num_shards=1, the single shard should equal the original."""
    weights = {"w": np.random.randn(5, 3).astype(np.float32)}
    shards, metadata = shard_weights(weights, num_shards=1)

    assert len(shards) == 1
    np.testing.assert_array_equal(shards[0]["w"], weights["w"])


def test_metadata_stores_original_shape():
    """Metadata should record the shape before any padding."""
    weights = {"w": np.random.randn(7, 4).astype(np.float32)}
    _, metadata = shard_weights(weights, num_shards=3)
    assert metadata["w"] == (7, 4)


def test_1d_tensor():
    """1D tensors should shard and gather correctly."""
    weights = {"bias": np.arange(10).astype(np.float32)}
    shards, metadata = shard_weights(weights, num_shards=3)
    # 10 -> pad to 12 -> 3 shards of 4
    assert shards[0]["bias"].shape == (4,)
    recovered = gather_weights(shards, metadata)
    np.testing.assert_array_equal(recovered["bias"], weights["bias"])
