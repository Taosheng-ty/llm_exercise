"""
Tests for Exercise 01: All-Reduce Simulation
"""

import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

simulate_all_reduce = _mod.simulate_all_reduce


def test_sum_basic():
    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = simulate_all_reduce(tensors, "sum")
    assert len(result) == 2
    expected = torch.tensor([4.0, 6.0])
    for r in result:
        assert torch.allclose(r, expected)


def test_mean_basic():
    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = simulate_all_reduce(tensors, "mean")
    expected = torch.tensor([2.0, 3.0])
    for r in result:
        assert torch.allclose(r, expected)


def test_max_basic():
    tensors = [
        torch.tensor([1.0, 5.0]),
        torch.tensor([3.0, 2.0]),
        torch.tensor([2.0, 4.0]),
    ]
    result = simulate_all_reduce(tensors, "max")
    expected = torch.tensor([3.0, 5.0])
    assert len(result) == 3
    for r in result:
        assert torch.allclose(r, expected)


def test_all_workers_identical():
    """After all-reduce, every worker must hold the same tensor."""
    tensors = [torch.randn(4, 3) for _ in range(5)]
    for op in ["sum", "mean", "max"]:
        result = simulate_all_reduce(tensors, op)
        for i in range(1, len(result)):
            assert torch.allclose(result[0], result[i])


def test_sum_multidim():
    tensors = [torch.ones(2, 3) * i for i in range(4)]
    result = simulate_all_reduce(tensors, "sum")
    # 0 + 1 + 2 + 3 = 6
    expected = torch.ones(2, 3) * 6.0
    for r in result:
        assert torch.allclose(r, expected)


def test_single_worker():
    tensors = [torch.tensor([7.0, 8.0])]
    for op in ["sum", "mean", "max"]:
        result = simulate_all_reduce(tensors, op)
        assert len(result) == 1
        assert torch.allclose(result[0], tensors[0])


def test_invalid_op_raises():
    tensors = [torch.tensor([1.0])]
    try:
        simulate_all_reduce(tensors, "min")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_clones_are_independent():
    """Modifying one worker's result should not affect others."""
    tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    result = simulate_all_reduce(tensors, "sum")
    result[0][0] = 999.0
    assert result[1][0] != 999.0
