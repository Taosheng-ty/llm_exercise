"""
Exercise 02: Model Weight Sharding and Gathering - Solution
"""

import numpy as np


def shard_weights(
    weights: dict[str, np.ndarray],
    num_shards: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, tuple]]:
    """Split weight tensors into shards along axis 0.

    Args:
        weights: Dict mapping parameter name to numpy array.
        num_shards: Number of shards to split into.

    Returns:
        A tuple of:
          - List of num_shards dicts, each mapping name -> shard array.
          - Metadata dict mapping name -> original shape tuple.
    """
    shard_dicts = [{} for _ in range(num_shards)]
    metadata = {}

    for name, tensor in weights.items():
        original_shape = tensor.shape
        metadata[name] = original_shape

        # Handle 0-d or 1-d arrays
        if tensor.ndim == 0:
            # Scalar: replicate to all shards
            for i in range(num_shards):
                shard_dicts[i][name] = tensor.copy()
            continue

        dim0 = tensor.shape[0]
        remainder = dim0 % num_shards
        if remainder != 0:
            pad_amount = num_shards - remainder
            pad_widths = [(0, pad_amount)] + [(0, 0)] * (tensor.ndim - 1)
            tensor = np.pad(tensor, pad_widths, mode="constant", constant_values=0)

        # Split along axis 0
        chunks = np.split(tensor, num_shards, axis=0)
        for i, chunk in enumerate(chunks):
            shard_dicts[i][name] = chunk

    return shard_dicts, metadata


def gather_weights(
    shards: list[dict[str, np.ndarray]],
    metadata: dict[str, tuple],
) -> dict[str, np.ndarray]:
    """Reconstruct original weight tensors from shards.

    Args:
        shards: List of shard dicts (output of shard_weights).
        metadata: Metadata dict with original shapes (output of shard_weights).

    Returns:
        Dict mapping parameter name to reconstructed numpy array,
        matching the original weights exactly.
    """
    result = {}

    for name, original_shape in metadata.items():
        if len(original_shape) == 0:
            # Scalar: just take the first shard's copy
            result[name] = shards[0][name].copy()
            continue

        # Concatenate all shards along axis 0
        chunks = [shard[name] for shard in shards]
        gathered = np.concatenate(chunks, axis=0)

        # Trim back to original shape (removes padding)
        slices = [slice(0, s) for s in original_shape]
        result[name] = gathered[tuple(slices)]

    return result
