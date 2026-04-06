"""Tests for Exercise 02: Experience Replay Buffer."""

import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ReplayBuffer = _mod.ReplayBuffer
Experience = _mod.Experience


def _make_exp(reward: float, idx: int = 0) -> Experience:
    return Experience(prompt=f"p{idx}", response=f"r{idx}", reward=reward)


def test_add_and_len():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_exp(1.0, i) for i in range(5)])
    assert len(buf) == 5
    assert not buf.is_full


def test_fifo_eviction():
    buf = ReplayBuffer(capacity=4)
    buf.add([_make_exp(float(i), i) for i in range(3)])
    evicted = buf.add([_make_exp(float(i), i) for i in range(3, 6)])
    assert evicted == 2
    assert len(buf) == 4
    # The oldest (0, 1) should have been evicted
    prompts = [e.prompt for e in buf.buffer]
    assert "p0" not in prompts
    assert "p1" not in prompts


def test_is_full():
    buf = ReplayBuffer(capacity=3)
    buf.add([_make_exp(1.0, i) for i in range(3)])
    assert buf.is_full


def test_uniform_sample():
    buf = ReplayBuffer(capacity=100, seed=0)
    buf.add([_make_exp(float(i), i) for i in range(50)])
    batch = buf.sample(10, prioritized=False)
    assert len(batch) == 10
    assert all(isinstance(e, Experience) for e in batch)


def test_priority_sample_bias():
    """High-reward items should be sampled more often with priority sampling."""
    buf = ReplayBuffer(capacity=100, seed=42)
    # 1 high-reward item, 99 low-reward items
    buf.add([_make_exp(0.0, i) for i in range(99)])
    buf.add([_make_exp(100.0, 99)])
    samples = buf.sample(1000, prioritized=True)
    high_count = sum(1 for s in samples if s.reward == 100.0)
    # Should be sampled much more than 1% (its uniform share = 1/100 = 1%)
    assert high_count > 50, f"Expected significant bias, got {high_count}/1000"


def test_sample_empty_raises():
    buf = ReplayBuffer(capacity=5)
    with pytest.raises(ValueError):
        buf.sample(1)


def test_clear():
    buf = ReplayBuffer(capacity=10)
    buf.add([_make_exp(1.0, i) for i in range(5)])
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_full


def test_sample_with_replacement():
    """When batch_size > buffer size, sampling with replacement still works."""
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.add([_make_exp(1.0, 0)])
    batch = buf.sample(10)
    assert len(batch) == 10
    assert all(e.prompt == "p0" for e in batch)
