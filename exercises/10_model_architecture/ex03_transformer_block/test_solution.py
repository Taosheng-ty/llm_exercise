"""Tests for Exercise 03: Complete Transformer Decoder Block"""

import importlib.util
import os
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TransformerBlock = _mod.TransformerBlock
CausalSelfAttention = _mod.CausalSelfAttention
RMSNorm = _mod.RMSNorm
SwiGLUFFN = _mod.SwiGLUFFN


def test_block_output_shape():
    block = TransformerBlock(dim=64, n_heads=4, ffn_hidden_dim=128)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64), f"Expected (2, 10, 64), got {out.shape}"


def test_causal_masking():
    """Future tokens should NOT influence past tokens."""
    torch.manual_seed(42)
    attn = CausalSelfAttention(dim=32, n_heads=4)

    # Two sequences: same prefix, different suffix
    x1 = torch.randn(1, 5, 32)
    x2 = x1.clone()
    x2[0, 3:] = torch.randn(2, 32)  # Change tokens at positions 3, 4

    out1 = attn(x1)
    out2 = attn(x2)

    # Outputs at positions 0, 1, 2 should be identical (causal = can't see future)
    for pos in range(3):
        assert torch.allclose(out1[0, pos], out2[0, pos], atol=1e-5), (
            f"Position {pos} should not be affected by future tokens"
        )

    # Position 3 should differ (it sees position 3 which changed)
    assert not torch.allclose(out1[0, 3], out2[0, 3], atol=1e-3), (
        "Position 3 should differ since its input changed"
    )


def test_residual_connection():
    """Output should contain the input signal (residual)."""
    torch.manual_seed(42)
    block = TransformerBlock(dim=32, n_heads=4, ffn_hidden_dim=64)
    x = torch.randn(1, 4, 32)
    out = block(x)
    # The output should not be identical to input (sublayers do something)
    assert not torch.allclose(out, x, atol=1e-3), "Block should transform input"
    # But should be correlated (residual connection preserves signal)
    # Check that the difference is not too large relative to input
    diff_norm = (out - x).norm()
    x_norm = x.norm()
    assert diff_norm < x_norm * 10, "Residual connection should keep output close to input"


def test_pre_norm_architecture():
    """Verify the block uses pre-norm (norm before sublayer, not after)."""
    block = TransformerBlock(dim=32, n_heads=4, ffn_hidden_dim=64)
    assert hasattr(block, 'attention_norm'), "Should have attention_norm"
    assert hasattr(block, 'ffn_norm'), "Should have ffn_norm"
    assert isinstance(block.attention_norm, RMSNorm)
    assert isinstance(block.ffn_norm, RMSNorm)


def test_gradient_flows():
    block = TransformerBlock(dim=32, n_heads=4, ffn_hidden_dim=64)
    x = torch.randn(2, 6, 32, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().mean() > 1e-6


def test_attention_no_bias():
    attn = CausalSelfAttention(dim=32, n_heads=4)
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None


def test_single_token():
    """Should work with sequence length 1."""
    block = TransformerBlock(dim=32, n_heads=4, ffn_hidden_dim=64)
    x = torch.randn(1, 1, 32)
    out = block(x)
    assert out.shape == (1, 1, 32)
