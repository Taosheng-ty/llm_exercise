"""
Solution for Exercise 05: Verify Tensor Integrity with Hashing
"""

import numpy as np


def compute_tensor_hash(tensor: np.ndarray) -> int:
    """Compute a simple hash of a numpy tensor."""
    raw_bytes = tensor.tobytes()

    # Pad to multiple of 4 bytes
    remainder = len(raw_bytes) % 4
    if remainder != 0:
        raw_bytes = raw_bytes + b"\x00" * (4 - remainder)

    # Interpret as uint32 and sum with uint64 to avoid overflow
    uint32_view = np.frombuffer(raw_bytes, dtype=np.uint32)
    hash_val = np.sum(uint32_view, dtype=np.uint64)

    return int(hash_val)


def save_checkpoint(
    filepath: str,
    tensors: dict[str, np.ndarray],
) -> dict[str, int]:
    """Save tensors to a .npz file and return their hashes."""
    np.savez(filepath, **tensors)

    hashes = {}
    for name, tensor in tensors.items():
        hashes[name] = compute_tensor_hash(tensor)

    return hashes


def load_checkpoint(
    filepath: str,
    expected_hashes: dict[str, int],
) -> dict[str, np.ndarray]:
    """Load tensors from a .npz file and verify their integrity."""
    data = np.load(filepath)

    tensors = {}
    for name in expected_hashes:
        if name not in data:
            raise ValueError(f"Missing tensor: {name}")

        tensor = data[name]
        actual_hash = compute_tensor_hash(tensor)

        if actual_hash != expected_hashes[name]:
            raise ValueError(
                f"Hash mismatch for '{name}': expected {expected_hashes[name]}, "
                f"got {actual_hash}"
            )

        tensors[name] = tensor

    return tensors
