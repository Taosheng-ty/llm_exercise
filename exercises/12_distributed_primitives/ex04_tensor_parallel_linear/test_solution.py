"""
Tests for Exercise 04: Tensor Parallel Linear Layer
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

column_parallel_linear = _mod.column_parallel_linear
row_parallel_linear = _mod.row_parallel_linear


def _reference_linear(input, weight, bias):
    out = input @ weight.t()
    if bias is not None:
        out = out + bias
    return out


def test_column_parallel_matches_linear_no_bias():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    w = torch.randn(6, 8)
    expected = _reference_linear(x, w, None)
    result = column_parallel_linear(x, w, None, num_shards=3)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-5)


def test_column_parallel_matches_linear_with_bias():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w = torch.randn(6, 8)
    b = torch.randn(6)
    expected = _reference_linear(x, w, b)
    result = column_parallel_linear(x, w, b, num_shards=2)
    assert torch.allclose(result, expected, atol=1e-5)


def test_row_parallel_matches_linear_no_bias():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    w = torch.randn(6, 8)
    expected = _reference_linear(x, w, None)
    result = row_parallel_linear(x, w, None, num_shards=4)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-5)


def test_row_parallel_matches_linear_with_bias():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w = torch.randn(6, 8)
    b = torch.randn(6)
    expected = _reference_linear(x, w, b)
    result = row_parallel_linear(x, w, b, num_shards=2)
    assert torch.allclose(result, expected, atol=1e-5)


def test_column_parallel_single_shard():
    """With 1 shard, result should be identical to standard linear."""
    torch.manual_seed(42)
    x = torch.randn(3, 10)
    w = torch.randn(5, 10)
    b = torch.randn(5)
    expected = _reference_linear(x, w, b)
    result = column_parallel_linear(x, w, b, num_shards=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_row_parallel_single_shard():
    torch.manual_seed(42)
    x = torch.randn(3, 10)
    w = torch.randn(5, 10)
    b = torch.randn(5)
    expected = _reference_linear(x, w, b)
    result = row_parallel_linear(x, w, b, num_shards=1)
    assert torch.allclose(result, expected, atol=1e-5)


def test_column_parallel_output_shape():
    x = torch.randn(2, 16)
    w = torch.randn(8, 16)
    result = column_parallel_linear(x, w, None, num_shards=4)
    assert result.shape == (2, 8)


def test_row_parallel_output_shape():
    x = torch.randn(2, 16)
    w = torch.randn(8, 16)
    result = row_parallel_linear(x, w, None, num_shards=4)
    assert result.shape == (2, 8)


def test_matches_nn_linear():
    """Both column and row parallel should match nn.Linear exactly."""
    torch.manual_seed(42)
    linear = nn.Linear(12, 6)
    x = torch.randn(5, 12)

    expected = linear(x)
    col_result = column_parallel_linear(x, linear.weight, linear.bias, num_shards=3)
    row_result = row_parallel_linear(x, linear.weight, linear.bias, num_shards=3)

    assert torch.allclose(col_result, expected, atol=1e-5)
    assert torch.allclose(row_result, expected, atol=1e-5)
