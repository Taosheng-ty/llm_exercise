"""
Exercise 02: Model Weight Sharding and Gathering

In distributed training, model weights must be sharded across multiple workers
(e.g., tensor parallelism) and later gathered back for checkpointing or weight
updates. This mirrors the pattern in slime's update_weight/ module where weights
are broadcast/gathered across NCCL groups.

Implement two functions:

1. shard_weights(weights, num_shards):
   - Takes a dict of {name: numpy_array} and splits each tensor along axis 0
     into num_shards pieces.
   - If a tensor's dim-0 size doesn't divide evenly by num_shards, pad with
     zeros at the end to make it divisible, then split.
   - Returns a list of num_shards dicts, where each dict has the same keys
     and the shard as the value.
   - Also returns a metadata dict mapping name -> original_shape (before padding).

2. gather_weights(shards, metadata):
   - Takes the list of shard dicts and metadata from shard_weights.
   - Concatenates shards along axis 0, then trims back to the original shape
     using metadata.
   - Returns a dict of {name: numpy_array} matching the original weights exactly.
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
    # TODO: Implement weight sharding
    raise NotImplementedError("Implement shard_weights")


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
    # TODO: Implement weight gathering
    raise NotImplementedError("Implement gather_weights")
