"""
Tests for Exercise 06: Data Parallel Partitioning
"""

import importlib.util
import os

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

partition_data = _mod.partition_data


def test_contiguous_even():
    result = partition_data(12, 3, "contiguous")
    assert result == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_contiguous_uneven():
    result = partition_data(10, 3, "contiguous")
    # 10 / 3 = 3 remainder 1, so rank 0 gets 4, ranks 1-2 get 3
    assert len(result[0]) == 4
    assert len(result[1]) == 3
    assert len(result[2]) == 3
    all_indices = sorted(sum(result, []))
    assert all_indices == list(range(10))


def test_interleaved():
    result = partition_data(9, 3, "interleaved")
    assert result[0] == [0, 3, 6]
    assert result[1] == [1, 4, 7]
    assert result[2] == [2, 5, 8]


def test_interleaved_uneven():
    result = partition_data(7, 3, "interleaved")
    assert result[0] == [0, 3, 6]
    assert result[1] == [1, 4]
    assert result[2] == [2, 5]


def test_all_indices_covered():
    """Every index must appear exactly once across all partitions."""
    for mode in ["contiguous", "interleaved"]:
        for ds in [7, 10, 13, 16]:
            for nr in [1, 2, 3, 4]:
                result = partition_data(ds, nr, mode)
                assert len(result) == nr
                all_idx = sorted(sum(result, []))
                assert all_idx == list(range(ds)), f"Failed for ds={ds}, nr={nr}, mode={mode}"


def test_balanced_basic():
    # lengths: [10, 1, 1, 1, 1, 1, 1, 1, 1, 1] => total = 19
    # Balanced across 2 ranks: rank with 10 should get ~10, other should get ~9
    lengths = [10, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = partition_data(10, 2, "balanced", lengths)
    totals = [sum(lengths[i] for i in part) for part in result]
    assert abs(totals[0] - totals[1]) <= 1  # Should be 10 vs 9


def test_balanced_all_indices_covered():
    lengths = [5, 3, 8, 2, 7, 1]
    result = partition_data(6, 3, "balanced", lengths)
    all_idx = sorted(sum(result, []))
    assert all_idx == list(range(6))


def test_balanced_sorted_ascending():
    """Each rank's indices should be in ascending order."""
    lengths = [5, 3, 8, 2, 7, 1, 4, 6]
    result = partition_data(8, 3, "balanced", lengths)
    for part in result:
        assert part == sorted(part)


def test_single_rank():
    result = partition_data(5, 1, "contiguous")
    assert result == [[0, 1, 2, 3, 4]]


def test_balanced_requires_lengths():
    try:
        partition_data(5, 2, "balanced", None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_contiguous_more_ranks_than_data():
    result = partition_data(2, 5, "contiguous")
    assert len(result) == 5
    all_idx = sorted(sum(result, []))
    assert all_idx == [0, 1]
    # 3 ranks should be empty
    empty_count = sum(1 for p in result if len(p) == 0)
    assert empty_count == 3
