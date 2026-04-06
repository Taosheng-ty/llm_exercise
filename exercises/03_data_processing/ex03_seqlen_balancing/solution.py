"""
Exercise 03: Sequence Length Balancing with Karmarkar-Karp - Solution
"""

import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int) -> list[list[int]]:
    """Partition sequences into k groups to minimize the max total length."""

    class Partition:
        """Represents one partition: a sum and a list of (index, value) items."""
        def __init__(self):
            self.sum = 0
            self.items: list[tuple[int, int]] = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other: "Partition"):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other: "Partition"):
            if self.sum != other.sum:
                return self.sum < other.sum
            return len(self.items) < len(other.items)

    class State:
        """A state holds k partitions sorted by sum in decreasing order."""
        def __init__(self, k: int):
            self.k = k
            self.partitions = [Partition() for _ in range(k)]

        @property
        def spread(self) -> int:
            return self.partitions[0].sum - self.partitions[-1].sum

        def sort_partitions(self):
            self.partitions.sort(reverse=True)

        def merge(self, other: "State"):
            """Merge: pair partition i of self with partition (k-1-i) of other."""
            for i in range(self.k):
                self.partitions[i].merge(other.partitions[self.k - 1 - i])
            self.sort_partitions()

        def get_index_lists(self) -> list[list[int]]:
            result = []
            for p in self.partitions:
                result.append(sorted(idx for idx, _ in p.items))
            return result

        def __lt__(self, other: "State"):
            # Max-heap by spread: state with LARGER spread should come first
            # Since heapq is a min-heap, invert the comparison
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.partitions[0] > other.partitions[0]

    # Sort sequences by value (ascending) to process smallest first
    sorted_seqs = sorted((val, idx) for idx, val in enumerate(seqlen_list))

    heap = []
    for val, idx in sorted_seqs:
        state = State(k_partitions)
        state.partitions[0].add(idx, val)
        state.sort_partitions()
        heapq.heappush(heap, state)

    while len(heap) > 1:
        s0 = heapq.heappop(heap)
        s1 = heapq.heappop(heap)
        s0.merge(s1)
        heapq.heappush(heap, s0)

    return heap[0].get_index_lists()


def get_partition_sums(seqlen_list: list[int], partitions: list[list[int]]) -> list[int]:
    """Compute the total sequence length for each partition."""
    return [sum(seqlen_list[idx] for idx in part) for part in partitions]


def partition_spread(seqlen_list: list[int], partitions: list[list[int]]) -> int:
    """Compute the spread (max_sum - min_sum) across partitions."""
    sums = get_partition_sums(seqlen_list, partitions)
    return max(sums) - min(sums)
