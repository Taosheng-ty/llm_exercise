import importlib.util
import os
import tempfile

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_tensor_hash = _mod.compute_tensor_hash
save_checkpoint = _mod.save_checkpoint
load_checkpoint = _mod.load_checkpoint


class TestComputeTensorHash:
    def test_deterministic(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        h1 = compute_tensor_hash(arr)
        h2 = compute_tensor_hash(arr)
        assert h1 == h2

    def test_different_data_different_hash(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        assert compute_tensor_hash(a) != compute_tensor_hash(b)

    def test_returns_int(self):
        arr = np.zeros(10, dtype=np.float32)
        h = compute_tensor_hash(arr)
        assert isinstance(h, int)

    def test_zeros(self):
        arr = np.zeros(8, dtype=np.float32)
        assert compute_tensor_hash(arr) == 0

    def test_uint8_not_divisible_by_4(self):
        """Test padding for arrays whose byte count is not divisible by 4."""
        arr = np.array([1, 2, 3], dtype=np.uint8)  # 3 bytes, needs padding
        h = compute_tensor_hash(arr)
        assert isinstance(h, int)
        assert h > 0

    def test_various_dtypes(self):
        for dtype in [np.float32, np.float64, np.int32, np.int16]:
            arr = np.arange(10, dtype=dtype)
            h = compute_tensor_hash(arr)
            assert isinstance(h, int)


class TestSaveLoadCheckpoint:
    def test_roundtrip(self):
        tensors = {
            "weight": np.random.randn(32, 32).astype(np.float32),
            "bias": np.random.randn(32).astype(np.float32),
        }
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            hashes = save_checkpoint(filepath, tensors)
            loaded = load_checkpoint(filepath, hashes)

            for name in tensors:
                assert np.array_equal(tensors[name], loaded[name])
        finally:
            os.unlink(filepath)

    def test_hash_mismatch_raises(self):
        tensors = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            hashes = save_checkpoint(filepath, tensors)
            # Corrupt the expected hash
            bad_hashes = {"w": hashes["w"] + 1}
            try:
                load_checkpoint(filepath, bad_hashes)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Hash mismatch" in str(e)
        finally:
            os.unlink(filepath)

    def test_missing_tensor_raises(self):
        tensors = {"w": np.array([1.0], dtype=np.float32)}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            save_checkpoint(filepath, tensors)
            try:
                load_checkpoint(filepath, {"nonexistent": 123})
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Missing" in str(e)
        finally:
            os.unlink(filepath)

    def test_multiple_tensors(self):
        tensors = {f"layer_{i}": np.random.randn(16, 16).astype(np.float32) for i in range(5)}
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            filepath = f.name

        try:
            hashes = save_checkpoint(filepath, tensors)
            assert len(hashes) == 5
            loaded = load_checkpoint(filepath, hashes)
            assert len(loaded) == 5
        finally:
            os.unlink(filepath)
