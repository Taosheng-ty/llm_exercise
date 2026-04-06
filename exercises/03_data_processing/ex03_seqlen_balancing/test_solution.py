"""Tests for Exercise 03: Sequence Length Balancing"""

import pytest
from .solution import karmarkar_karp, get_partition_sums, partition_spread


class TestKarmarkarKarp:
    def test_two_partitions_balanced(self):
        """Equal total is achievable: [10, 8] vs [7, 6, 5] = 18 vs 18."""
        lengths = [10, 8, 7, 6, 5]
        partitions = karmarkar_karp(lengths, k_partitions=2)
        assert len(partitions) == 2
        # All indices covered
        all_indices = sorted(idx for part in partitions for idx in part)
        assert all_indices == list(range(5))
        # Check balance: spread should be small (KK gets spread <= 2 here)
        sums = get_partition_sums(lengths, partitions)
        assert max(sums) - min(sums) <= 2  # near-perfect balance

    def test_three_partitions(self):
        lengths = [10, 9, 8, 7, 6, 5]
        partitions = karmarkar_karp(lengths, k_partitions=3)
        assert len(partitions) == 3
        all_indices = sorted(idx for part in partitions for idx in part)
        assert all_indices == list(range(6))
        # Total is 45, ideal per partition is 15
        sums = get_partition_sums(lengths, partitions)
        assert max(sums) - min(sums) <= 2  # good balance

    def test_single_partition(self):
        lengths = [5, 3, 7]
        partitions = karmarkar_karp(lengths, k_partitions=1)
        assert len(partitions) == 1
        assert sorted(partitions[0]) == [0, 1, 2]

    def test_equal_lengths(self):
        lengths = [10, 10, 10, 10]
        partitions = karmarkar_karp(lengths, k_partitions=2)
        sums = get_partition_sums(lengths, partitions)
        assert sums[0] == sums[1] == 20

    def test_all_indices_present(self):
        """Every index should appear exactly once across all partitions."""
        lengths = [1, 2, 3, 4, 5, 6, 7, 8]
        partitions = karmarkar_karp(lengths, k_partitions=4)
        all_indices = sorted(idx for part in partitions for idx in part)
        assert all_indices == list(range(8))
        # No duplicates
        flat = [idx for part in partitions for idx in part]
        assert len(flat) == len(set(flat))

    def test_better_than_naive(self):
        """KK should produce better balance than naive sequential assignment."""
        lengths = [100, 90, 80, 70, 60, 50, 40, 30]
        partitions = karmarkar_karp(lengths, k_partitions=4)
        kk_spread = partition_spread(lengths, partitions)
        # Naive: just split in order -> [100,90], [80,70], [60,50], [40,30]
        # = 190, 150, 110, 70 -> spread = 120
        # KK should do much better
        assert kk_spread < 120


class TestGetPartitionSums:
    def test_basic(self):
        lengths = [3, 5, 2, 8]
        partitions = [[0, 2], [1, 3]]
        sums = get_partition_sums(lengths, partitions)
        assert sums == [5, 13]

    def test_single_element_partitions(self):
        lengths = [10, 20, 30]
        partitions = [[0], [1], [2]]
        sums = get_partition_sums(lengths, partitions)
        assert sums == [10, 20, 30]


class TestPartitionSpread:
    def test_balanced(self):
        lengths = [5, 5]
        partitions = [[0], [1]]
        assert partition_spread(lengths, partitions) == 0

    def test_unbalanced(self):
        lengths = [10, 1]
        partitions = [[0], [1]]
        assert partition_spread(lengths, partitions) == 9
