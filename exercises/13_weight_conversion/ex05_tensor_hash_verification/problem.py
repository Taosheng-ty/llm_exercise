"""
Exercise 05: Verify Tensor Integrity with Hashing (Easy, numpy)

When saving and loading checkpoints, we need to verify that tensor data has not
been corrupted. A simple approach is to compute a hash of the tensor data and
compare before/after.

The hash method (inspired by slime tensor_backper.py patterns):
  1. View the tensor as raw bytes (uint8)
  2. Reinterpret as uint32 (pad with zeros if needed to make length divisible by 4)
  3. Sum all uint32 values (use uint64 to avoid overflow)
  4. Return the sum as an integer

Reference: slime/utils/tensor_backper.py backup/restore pattern

Tasks:
    1. Implement compute_tensor_hash() using numpy.
    2. Implement verify_checkpoint() that checks all tensors match their saved hashes.
    3. Implement save_checkpoint() / load_checkpoint() using np.savez and hash verification.
"""

import numpy as np


def compute_tensor_hash(tensor: np.ndarray) -> int:
    """Compute a simple hash of a numpy tensor.

    Steps:
      1. Get raw bytes via tensor.tobytes()
      2. Pad to multiple of 4 bytes with zeros
      3. Interpret as uint32 array
      4. Sum all values using uint64 accumulator
      5. Return as Python int

    Args:
        tensor: any numpy array

    Returns:
        Integer hash value
    """
    # TODO: Implement this function
    raise NotImplementedError


def save_checkpoint(
    filepath: str,
    tensors: dict[str, np.ndarray],
) -> dict[str, int]:
    """Save tensors to a .npz file and return their hashes.

    Args:
        filepath: path to save the .npz file
        tensors: dict mapping name -> numpy array

    Returns:
        dict mapping name -> hash value
    """
    # TODO: Implement this function
    # Hint: use np.savez(filepath, **tensors)
    raise NotImplementedError


def load_checkpoint(
    filepath: str,
    expected_hashes: dict[str, int],
) -> dict[str, np.ndarray]:
    """Load tensors from a .npz file and verify their integrity.

    Args:
        filepath: path to the .npz file
        expected_hashes: dict mapping name -> expected hash value

    Returns:
        dict mapping name -> numpy array

    Raises:
        ValueError: if any tensor hash does not match expected
    """
    # TODO: Implement this function
    raise NotImplementedError
