"""
Solution for Exercise 04: Tensor Parallel Linear Layer
"""

import torch


def column_parallel_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_shards: int,
) -> torch.Tensor:
    """
    Simulate column-parallel linear: split weight along output dim.

    Weight shape: (out_features, in_features)
    Split along dim=0 (output features) into num_shards chunks.
    Each shard computes: input @ weight_shard.T (+ bias_shard)
    All-gather: concatenate shard outputs along the feature dim.
    """
    # Split weight along output dimension (dim=0)
    weight_shards = torch.chunk(weight, num_shards, dim=0)

    # Split bias along output dimension if present
    bias_shards = (
        torch.chunk(bias, num_shards, dim=0) if bias is not None else [None] * num_shards
    )

    # Each shard computes its portion of the output
    shard_outputs = []
    for w_shard, b_shard in zip(weight_shards, bias_shards):
        out = input @ w_shard.t()
        if b_shard is not None:
            out = out + b_shard
        shard_outputs.append(out)

    # All-gather: concatenate along output feature dimension
    return torch.cat(shard_outputs, dim=-1)


def row_parallel_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_shards: int,
) -> torch.Tensor:
    """
    Simulate row-parallel linear: split weight along input dim.

    Weight shape: (out_features, in_features)
    Split along dim=1 (input features) into num_shards chunks.
    Input is also split along the feature dimension.
    Each shard computes: input_shard @ weight_shard.T
    All-reduce: sum all shard outputs element-wise, then add bias.
    """
    # Split weight along input dimension (dim=1)
    weight_shards = torch.chunk(weight, num_shards, dim=1)

    # Split input along feature dimension
    input_shards = torch.chunk(input, num_shards, dim=-1)

    # Each shard computes its partial output
    shard_outputs = []
    for x_shard, w_shard in zip(input_shards, weight_shards):
        out = x_shard @ w_shard.t()
        shard_outputs.append(out)

    # All-reduce: sum partial outputs
    result = torch.stack(shard_outputs).sum(dim=0)

    # Bias is added after the all-reduce (not sharded)
    if bias is not None:
        result = result + bias

    return result
