"""
Solution for Exercise 06: Data Parallel Partitioning
"""

import numpy as np


def partition_data(
    dataset_size: int,
    num_ranks: int,
    mode: str = "contiguous",
    lengths: list[int] | None = None,
) -> list[list[int]]:
    """
    Partition dataset indices across data-parallel ranks.
    """
    if mode == "contiguous":
        base = dataset_size // num_ranks
        remainder = dataset_size % num_ranks
        partitions = []
        start = 0
        for r in range(num_ranks):
            size = base + (1 if r < remainder else 0)
            partitions.append(list(range(start, start + size)))
            start += size
        return partitions

    elif mode == "interleaved":
        partitions = [[] for _ in range(num_ranks)]
        for i in range(dataset_size):
            partitions[i % num_ranks].append(i)
        return partitions

    elif mode == "balanced":
        if lengths is None:
            raise ValueError("lengths must be provided for balanced mode")

        # Greedy: assign samples sorted by length (descending) to the rank
        # with smallest current total length
        sorted_indices = sorted(range(dataset_size), key=lambda i: -lengths[i])
        partitions = [[] for _ in range(num_ranks)]
        totals = np.zeros(num_ranks, dtype=np.int64)

        for idx in sorted_indices:
            # Assign to the rank with the smallest total
            rank = int(np.argmin(totals))
            partitions[rank].append(idx)
            totals[rank] += lengths[idx]

        # Sort each partition by index (ascending)
        for r in range(num_ranks):
            partitions[r].sort()
        return partitions

    else:
        raise ValueError(f"Unknown mode: {mode}")
