"""
Exercise 03: Sequence Length Balancing with Karmarkar-Karp (Hard)

In distributed LLM training, data is split across multiple GPUs (data parallel ranks).
If one GPU gets much longer sequences than others, it becomes a bottleneck and all
other GPUs wait idle. Sequence length balancing assigns sequences to partitions to
minimize the maximum total sequence length across partitions.

This is a variant of the multiway number partitioning problem. The Karmarkar-Karp
algorithm (Largest Differencing Method) provides a good approximation:

1. Create initial "states" - each containing one sequence.
2. Use a max-heap ordered by "spread" (difference between largest and smallest partition).
3. Repeatedly pop the two states with largest spread and merge them by pairing
   the largest partition of one with the smallest partition of the other.
4. Continue until one state remains, which gives the final partitioning.

Reference: slime/utils/seqlen_balancing.py karmarkar_karp()
"""

import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int) -> list[list[int]]:
    """Partition sequences into k groups to minimize the max total length.

    Implements the Largest Differencing Method (LDM) / Karmarkar-Karp algorithm
    for multiway number partitioning.

    Algorithm:
    1. For each sequence, create a State with k partitions where the sequence
       is placed in one partition and the rest are empty.
    2. Push all states into a max-heap (ordered by spread = max_sum - min_sum).
    3. While more than one state remains:
       a. Pop the two states with largest spread.
       b. Merge them: pair partition i of state0 with partition (k-1-i) of state1.
          This pairs the largest with the smallest to balance the sums.
       c. Re-sort the partitions by sum in decreasing order.
       d. Push the merged state back.
    4. Return the partition assignments from the final state.

    Args:
        seqlen_list: List of sequence lengths (positive integers).
        k_partitions: Number of partitions to create.

    Returns:
        A list of k lists, where each inner list contains the indices (into
        seqlen_list) of the sequences assigned to that partition. Every index
        from 0..len(seqlen_list)-1 must appear in exactly one partition.

    Worked example (k=2, sequences=[10, 8, 7, 6, 5]):
        Step 1: Create initial states (each has 2 partitions: [seq] and []):
          S0: partitions=([0],[])  sums=(10,0)  spread=10
          S1: partitions=([1],[])  sums=(8,0)   spread=8
          S2: partitions=([2],[])  sums=(7,0)   spread=7
          S3: partitions=([3],[])  sums=(6,0)   spread=6
          S4: partitions=([4],[])  sums=(5,0)   spread=5

        Step 2: Pop two largest-spread states (S0, S1). Merge by pairing
          partition 0 of S0 with partition 1 of S1, and vice versa:
          merged sums = (10+0, 0+8) = (10, 8). Re-sort descending -> (10, 8).
          New state: partitions=([0],[1]) sums=(10,8) spread=2

        Continue merging until one state remains. Final result gives a
        well-balanced partition like [10,8] vs [7,6,5] -> sums 18 vs 18.

    Example:
        >>> partitions = karmarkar_karp([10, 8, 7, 6, 5], k_partitions=2)
        >>> # One good split: [10, 8] vs [7, 6, 5] -> sums 18 vs 18
    """
    # TODO: Implement this function
    raise NotImplementedError


def get_partition_sums(seqlen_list: list[int], partitions: list[list[int]]) -> list[int]:
    """Compute the total sequence length for each partition.

    Args:
        seqlen_list: The original list of sequence lengths.
        partitions: List of lists of indices into seqlen_list.

    Returns:
        A list of sums, one per partition.
    """
    # TODO: Implement this function
    raise NotImplementedError


def partition_spread(seqlen_list: list[int], partitions: list[list[int]]) -> int:
    """Compute the spread (max_sum - min_sum) across partitions.

    Args:
        seqlen_list: The original list of sequence lengths.
        partitions: List of lists of indices into seqlen_list.

    Returns:
        The difference between the largest and smallest partition sums.
    """
    # TODO: Implement this function
    raise NotImplementedError
