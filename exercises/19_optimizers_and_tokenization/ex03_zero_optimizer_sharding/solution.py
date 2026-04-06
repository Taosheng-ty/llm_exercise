"""Solution for Exercise 03: ZeRO Optimizer Sharding"""

import numpy as np


def _split_into_shards(arr: np.ndarray, num_ranks: int) -> list[np.ndarray]:
    """Split an array into num_ranks shards. Last shard gets any remainder."""
    shard_size = len(arr) // num_ranks
    shards = []
    for i in range(num_ranks):
        start = i * shard_size
        if i == num_ranks - 1:
            # Last rank gets the remainder
            shards.append(arr[start:].copy())
        else:
            shards.append(arr[start : start + shard_size].copy())
    return shards


def shard_optimizer_state(
    state: dict[str, np.ndarray], num_ranks: int
) -> list[dict[str, np.ndarray]]:
    """Partition optimizer state across ranks (ZeRO Stage 1)."""
    m_shards = _split_into_shards(state["m"], num_ranks)
    v_shards = _split_into_shards(state["v"], num_ranks)
    return [{"m": m_shards[i], "v": v_shards[i]} for i in range(num_ranks)]


def gather_for_step(
    sharded_states: list[dict[str, np.ndarray]], rank: int
) -> dict[str, np.ndarray]:
    """All-gather optimizer state from all ranks to reconstruct full state."""
    m_full = np.concatenate([s["m"] for s in sharded_states])
    v_full = np.concatenate([s["v"] for s in sharded_states])
    return {"m": m_full, "v": v_full}


def reduce_scatter_gradients(
    grads: np.ndarray, num_ranks: int
) -> list[np.ndarray]:
    """Reduce-scatter gradients across ranks (ZeRO Stage 2)."""
    return _split_into_shards(grads, num_ranks)


def shard_parameters(params: np.ndarray, num_ranks: int) -> list[np.ndarray]:
    """Partition parameters across ranks (ZeRO Stage 3)."""
    return _split_into_shards(params, num_ranks)


def all_gather_params(sharded_params: list[np.ndarray]) -> np.ndarray:
    """All-gather parameters from all ranks."""
    return np.concatenate(sharded_params)
