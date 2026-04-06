"""Tests for Exercise 05: LoRA Linear Layer"""

import importlib.util
import os
import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LoRALinear = _mod.LoRALinear


def test_output_shape():
    linear = nn.Linear(32, 64)
    lora = LoRALinear(linear, r=4, alpha=1.0)
    x = torch.randn(2, 10, 32)
    out = lora(x)
    assert out.shape == (2, 10, 64)


def test_frozen_original_weights():
    linear = nn.Linear(16, 32)
    lora = LoRALinear(linear, r=4)
    assert not lora.linear.weight.requires_grad
    if lora.linear.bias is not None:
        assert not lora.linear.bias.requires_grad


def test_lora_params_trainable():
    linear = nn.Linear(16, 32)
    lora = LoRALinear(linear, r=4)
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert lora.lora_A.shape == (4, 16)
    assert lora.lora_B.shape == (32, 4)


def test_lora_b_init_zero():
    """B should be initialized to zeros so LoRA starts as identity."""
    linear = nn.Linear(16, 32)
    lora = LoRALinear(linear, r=4)
    assert torch.all(lora.lora_B == 0), "lora_B should be initialized to zeros"


def test_zero_init_matches_original():
    """With B=0, LoRA output should match original linear."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 32)
    x = torch.randn(2, 5, 16)
    original_out = linear(x).clone()

    lora = LoRALinear(linear, r=4)
    lora_out = lora(x)
    assert torch.allclose(lora_out, original_out, atol=1e-5), (
        "With B=0, LoRA should produce same output as original"
    )


def test_merge_unmerge():
    """After merge, output should be the same. After unmerge, should revert."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 32)
    lora = LoRALinear(linear, r=4, alpha=2.0)
    # Set B to nonzero so LoRA has an effect
    lora.lora_B.data.normal_()

    x = torch.randn(2, 5, 16)
    out_before = lora(x).clone()

    lora.merge()
    out_merged = lora(x)
    assert torch.allclose(out_before, out_merged, atol=1e-4), (
        "Output should be the same after merge"
    )

    lora.unmerge()
    out_unmerged = lora(x)
    assert torch.allclose(out_before, out_unmerged, atol=1e-4), (
        "Output should be the same after unmerge"
    )


def test_lora_changes_output():
    """Nonzero LoRA weights should change the output."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 32)
    x = torch.randn(2, 5, 16)
    original_out = linear(x).clone()

    lora = LoRALinear(linear, r=4, alpha=1.0)
    lora.lora_B.data.normal_()  # Make B nonzero
    lora_out = lora(x)
    assert not torch.allclose(lora_out, original_out, atol=1e-3), (
        "Nonzero LoRA should change the output"
    )


def test_gradient_only_to_lora():
    """Gradients should flow to LoRA params but not to frozen weights."""
    linear = nn.Linear(16, 32)
    lora = LoRALinear(linear, r=4)
    lora.lora_B.data.normal_()

    x = torch.randn(2, 5, 16)
    out = lora(x)
    out.sum().backward()

    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
