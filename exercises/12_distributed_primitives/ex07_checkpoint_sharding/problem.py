"""
Exercise 07: Checkpoint Sharding (Medium, numpy)

Shard and reconstruct a model state dict, similar to how HuggingFace
safetensors shards model weights across multiple files.

When a model is too large for a single file, its state dict is split across
multiple shard files with an index file that maps tensor names to their
shard file.

Implement the following functions:

    shard_state_dict(
        state_dict: dict[str, np.ndarray],
        num_shards: int,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, str]]

    reconstruct_state_dict(
        shards: list[dict[str, np.ndarray]],
        index: dict[str, str],
    ) -> dict[str, np.ndarray]

shard_state_dict:
    Args:
        state_dict: Mapping from parameter name to numpy array.
        num_shards: Number of shard files to create.
    Returns:
        (shards, index) where:
        - shards: list of num_shards dicts, each mapping param names to arrays.
          Tensors are assigned to shards to balance total size (number of elements).
          Use greedy assignment: iterate over tensors sorted by size (descending),
          assign each to the shard with the smallest current total size.
        - index: dict mapping each tensor name to its shard filename,
          using the pattern "shard_{i:05d}.bin" (0-indexed).

reconstruct_state_dict:
    Args:
        shards: The list of shard dicts (as returned by shard_state_dict).
        index: The index mapping (as returned by shard_state_dict).
    Returns:
        The reconstructed state_dict. Each tensor must be numerically identical
        to the original.
"""

import numpy as np


def shard_state_dict(
    state_dict: dict[str, np.ndarray],
    num_shards: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, str]]:
    """
    Shard a state dict across multiple files with balanced sizes.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement shard_state_dict")


def reconstruct_state_dict(
    shards: list[dict[str, np.ndarray]],
    index: dict[str, str],
) -> dict[str, np.ndarray]:
    """
    Reconstruct original state dict from shards and index.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement reconstruct_state_dict")
