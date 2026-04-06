"""
Solution for Exercise 07: Checkpoint Sharding
"""

import numpy as np


def shard_state_dict(
    state_dict: dict[str, np.ndarray],
    num_shards: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, str]]:
    """
    Shard a state dict across multiple files with balanced sizes.
    """
    # Sort tensors by size (descending) for greedy assignment
    sorted_names = sorted(state_dict.keys(), key=lambda n: -state_dict[n].size)

    shards = [{} for _ in range(num_shards)]
    shard_sizes = np.zeros(num_shards, dtype=np.int64)
    index = {}

    for name in sorted_names:
        tensor = state_dict[name]
        # Assign to the shard with the smallest current size
        shard_idx = int(np.argmin(shard_sizes))
        shards[shard_idx][name] = tensor
        shard_sizes[shard_idx] += tensor.size
        index[name] = f"shard_{shard_idx:05d}.bin"

    return shards, index


def reconstruct_state_dict(
    shards: list[dict[str, np.ndarray]],
    index: dict[str, str],
) -> dict[str, np.ndarray]:
    """
    Reconstruct original state dict from shards and index.
    """
    # Build a mapping from shard filename to shard index
    shard_name_to_idx = {}
    for i in range(len(shards)):
        shard_name_to_idx[f"shard_{i:05d}.bin"] = i

    result = {}
    for name, shard_file in index.items():
        shard_idx = shard_name_to_idx[shard_file]
        result[name] = shards[shard_idx][name]

    return result
