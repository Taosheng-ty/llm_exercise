"""Exercise 03: ZeRO Optimizer Sharding (Hard, PyTorch/numpy)

ZeRO (Zero Redundancy Optimizer) is a key technique from DeepSpeed that enables
training very large models by partitioning optimizer state, gradients, and
parameters across data-parallel ranks.

ZeRO Stage 1: Partition optimizer states (m, v in Adam) across ranks
    - Each rank only stores 1/N of the optimizer state
    - All ranks still hold full gradients and parameters

ZeRO Stage 2: + Partition gradients across ranks
    - Gradients are reduce-scattered so each rank only holds its shard
    - Each rank updates its shard of parameters using its shard of optimizer state

ZeRO Stage 3: + Partition parameters across ranks
    - Parameters are all-gathered only when needed for forward/backward
    - Each rank stores only 1/N of the parameters persistently

Implement the following simulation functions:
    - shard_optimizer_state(state, num_ranks) -> list of per-rank state dicts
    - gather_for_step(sharded_states, rank) -> reconstructed full state
    - reduce_scatter_gradients(grads, num_ranks) -> list of per-rank gradient shards
    - shard_parameters(params, num_ranks) -> list of per-rank parameter shards
    - all_gather_params(sharded_params) -> full parameter tensor
"""

import numpy as np


def shard_optimizer_state(
    state: dict[str, np.ndarray], num_ranks: int
) -> list[dict[str, np.ndarray]]:
    """Partition optimizer state across ranks (ZeRO Stage 1).

    Args:
        state: dict with 'm' (first moment) and 'v' (second moment) arrays,
               each of shape (num_params,)
        num_ranks: number of data-parallel ranks

    Returns:
        List of dicts, one per rank, each containing shards of 'm' and 'v'.
        If num_params is not evenly divisible, the last rank gets the remainder.
    """
    # TODO: split state['m'] and state['v'] into num_ranks shards
    raise NotImplementedError("Implement shard_optimizer_state")


def gather_for_step(
    sharded_states: list[dict[str, np.ndarray]], rank: int
) -> dict[str, np.ndarray]:
    """All-gather optimizer state from all ranks to reconstruct full state.

    This simulates the communication needed when a rank needs the full
    optimizer state (e.g., for debugging or checkpointing).

    Args:
        sharded_states: list of per-rank state dicts from shard_optimizer_state
        rank: the rank requesting the full state (unused in simulation, all get same result)

    Returns:
        dict with full 'm' and 'v' arrays reconstructed by concatenation.
    """
    # TODO: concatenate shards from all ranks
    raise NotImplementedError("Implement gather_for_step")


def reduce_scatter_gradients(
    grads: np.ndarray, num_ranks: int
) -> list[np.ndarray]:
    """Reduce-scatter gradients across ranks (ZeRO Stage 2).

    In real distributed training, each rank computes full gradients, then
    reduce-scatter sums them and distributes shards. Here we simulate
    a single-machine version: given the full gradient tensor (already summed),
    partition it into shards.

    Args:
        grads: gradient array of shape (num_params,), assumed already all-reduced
        num_ranks: number of ranks

    Returns:
        List of gradient shards, one per rank.
    """
    # TODO: split the gradient array into num_ranks chunks
    raise NotImplementedError("Implement reduce_scatter_gradients")


def shard_parameters(params: np.ndarray, num_ranks: int) -> list[np.ndarray]:
    """Partition parameters across ranks (ZeRO Stage 3).

    Args:
        params: flat parameter array of shape (num_params,)
        num_ranks: number of ranks

    Returns:
        List of parameter shards, one per rank.
    """
    # TODO: split parameters into num_ranks chunks
    raise NotImplementedError("Implement shard_parameters")


def all_gather_params(sharded_params: list[np.ndarray]) -> np.ndarray:
    """All-gather parameters from all ranks.

    Args:
        sharded_params: list of parameter shards from shard_parameters

    Returns:
        Full parameter array reconstructed by concatenation.
    """
    # TODO: concatenate all shards
    raise NotImplementedError("Implement all_gather_params")
