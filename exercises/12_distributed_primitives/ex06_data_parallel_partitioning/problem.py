"""
Exercise 06: Data Parallel Partitioning (Easy, numpy)

Partition a dataset across data-parallel ranks. In distributed data-parallel
training, each GPU (rank) processes a different subset of the data.

Reference: slime _split_train_data_by_dp() which supports both round-robin
and balanced-by-length partitioning.

Implement the following function:

    partition_data(
        dataset_size: int,
        num_ranks: int,
        mode: str = "contiguous",
        lengths: list[int] | None = None,
    ) -> list[list[int]]

Args:
    dataset_size: Total number of samples.
    num_ranks: Number of data-parallel ranks.
    mode: One of:
        - "contiguous": Each rank gets a contiguous block of indices.
          Rank 0 gets [0..k-1], rank 1 gets [k..2k-1], etc.
          If not evenly divisible, earlier ranks get one extra sample.
        - "interleaved": Round-robin assignment. Sample i goes to rank (i % num_ranks).
        - "balanced": Partition so that the total length per rank is as balanced
          as possible. Requires the `lengths` argument. Use a greedy approach:
          assign each sample (sorted by length descending) to the rank with the
          smallest current total length.
    lengths: Required when mode="balanced". A list of length dataset_size
        with the length/cost of each sample.

Returns:
    A list of num_ranks lists, where each inner list contains the sample
    indices assigned to that rank, in ascending order.
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

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement partition_data")
