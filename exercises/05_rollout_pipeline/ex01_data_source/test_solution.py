"""Tests for Exercise 01: Rollout Data Source."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RolloutDataSource = _mod.RolloutDataSource


def test_load_and_basic_batch():
    ds = RolloutDataSource(seed=0)
    ds.load_data([1, 2, 3, 4, 5])
    batch = ds.get_batch(3)
    assert len(batch) == 3
    assert ds.remaining_in_epoch == 2
    assert ds.current_epoch == 0


def test_epoch_increments_on_exhaustion():
    ds = RolloutDataSource(seed=0)
    ds.load_data([10, 20, 30, 40])
    ds.get_batch(4)
    assert ds.current_epoch == 0
    assert ds.remaining_in_epoch == 0
    # Next batch triggers new epoch
    ds.get_batch(2)
    assert ds.current_epoch == 1


def test_wrap_around_batch():
    """When a batch spans the epoch boundary, items come from two epochs."""
    ds = RolloutDataSource(seed=42)
    ds.load_data([1, 2, 3, 4, 5])
    ds.get_batch(3)  # offset=3, remaining=2
    batch = ds.get_batch(4)  # needs 4, only 2 left -> wrap
    assert len(batch) == 4
    assert ds.current_epoch == 1
    assert ds.offset == 2


def test_reproducible_shuffling():
    """Same seed + epoch produces same order."""
    ds1 = RolloutDataSource(seed=99)
    ds1.load_data(list(range(20)))
    b1 = ds1.get_batch(20)

    ds2 = RolloutDataSource(seed=99)
    ds2.load_data(list(range(20)))
    b2 = ds2.get_batch(20)

    assert b1 == b2


def test_different_epochs_different_order():
    ds = RolloutDataSource(seed=7)
    ds.load_data(list(range(10)))
    epoch0 = ds.get_batch(10)
    epoch1 = ds.get_batch(10)
    # Extremely unlikely to be identical with 10 items
    assert epoch0 != epoch1
    assert ds.current_epoch == 1


def test_reset():
    ds = RolloutDataSource(seed=0)
    ds.load_data(list(range(8)))
    ds.get_batch(8)
    ds.get_batch(4)
    assert ds.current_epoch == 1
    ds.reset()
    assert ds.current_epoch == 0
    assert ds.offset == 0
    assert ds.remaining_in_epoch == 8


def test_empty_data_raises():
    ds = RolloutDataSource()
    with pytest.raises(ValueError):
        ds.get_batch(1)


def test_batch_size_exceeds_data_raises():
    ds = RolloutDataSource()
    ds.load_data([1, 2, 3])
    with pytest.raises(ValueError):
        ds.get_batch(5)
