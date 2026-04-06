"""Tests for Exercise 04: Sinusoidal Positional Encoding"""

import importlib.util
import os
import math
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

sinusoidal_positional_encoding = _mod.sinusoidal_positional_encoding


def test_output_shape():
    pe = sinusoidal_positional_encoding(100, 64)
    assert pe.shape == (100, 64), f"Expected (100, 64), got {pe.shape}"


def test_values_in_range():
    pe = sinusoidal_positional_encoding(50, 32)
    assert pe.min() >= -1.0 and pe.max() <= 1.0, "Values should be in [-1, 1]"


def test_position_zero():
    """At position 0, sin columns should be 0 and cos columns should be 1."""
    pe = sinusoidal_positional_encoding(10, 16)
    assert torch.allclose(pe[0, 0::2], torch.zeros(8), atol=1e-6), "sin(0) should be 0"
    assert torch.allclose(pe[0, 1::2], torch.ones(8), atol=1e-6), "cos(0) should be 1"


def test_specific_values():
    """Check a few manually computed values."""
    pe = sinusoidal_positional_encoding(10, 8)
    # pos=1, dim=0: sin(1 / 10000^(0/8)) = sin(1)
    expected_sin = math.sin(1.0)
    assert abs(pe[1, 0].item() - expected_sin) < 1e-5, f"Expected sin(1)={expected_sin}, got {pe[1, 0].item()}"

    # pos=1, dim=1: cos(1 / 10000^(0/8)) = cos(1)
    expected_cos = math.cos(1.0)
    assert abs(pe[1, 1].item() - expected_cos) < 1e-5, f"Expected cos(1)={expected_cos}, got {pe[1, 1].item()}"


def test_different_positions_differ():
    pe = sinusoidal_positional_encoding(50, 32)
    assert not torch.allclose(pe[0], pe[1]), "Different positions should have different encodings"
    assert not torch.allclose(pe[5], pe[10]), "Different positions should have different encodings"


def test_deterministic():
    pe1 = sinusoidal_positional_encoding(20, 16)
    pe2 = sinusoidal_positional_encoding(20, 16)
    assert torch.allclose(pe1, pe2), "Positional encoding should be deterministic"
