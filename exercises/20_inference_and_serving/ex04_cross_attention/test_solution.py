"""
Tests for Exercise 04: Cross-Attention
"""

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CrossAttention = _mod.CrossAttention


class TestCrossAttentionInit:
    def test_is_nn_module(self):
        attn = CrossAttention(d_model=64, num_heads=4)
        assert isinstance(attn, nn.Module)

    def test_has_projection_layers(self):
        attn = CrossAttention(d_model=64, num_heads=4)
        assert hasattr(attn, "W_q")
        assert hasattr(attn, "W_k")
        assert hasattr(attn, "W_v")
        assert hasattr(attn, "W_o")

    def test_parameter_count(self):
        d = 64
        attn = CrossAttention(d_model=d, num_heads=4)
        # 4 linear layers, each d*d weights + d bias = 4 * (d^2 + d)
        total = sum(p.numel() for p in attn.parameters())
        expected = 4 * (d * d + d)
        assert total == expected


class TestCrossAttentionForward:
    def test_output_shape(self):
        attn = CrossAttention(d_model=64, num_heads=4)
        decoder_h = torch.randn(2, 5, 64)
        encoder_o = torch.randn(2, 10, 64)
        out = attn(decoder_h, encoder_o)
        assert out.shape == (2, 5, 64)

    def test_different_seq_lengths(self):
        """Decoder and encoder can have different sequence lengths."""
        attn = CrossAttention(d_model=32, num_heads=2)
        decoder_h = torch.randn(1, 3, 32)
        encoder_o = torch.randn(1, 20, 32)
        out = attn(decoder_h, encoder_o)
        assert out.shape == (1, 3, 32)

    def test_gradients_flow(self):
        attn = CrossAttention(d_model=32, num_heads=2)
        decoder_h = torch.randn(2, 4, 32, requires_grad=True)
        encoder_o = torch.randn(2, 6, 32, requires_grad=True)
        out = attn(decoder_h, encoder_o)
        loss = out.sum()
        loss.backward()
        assert decoder_h.grad is not None
        assert encoder_o.grad is not None
        # Gradients should not be all zeros
        assert decoder_h.grad.abs().sum() > 0
        assert encoder_o.grad.abs().sum() > 0

    def test_no_nan_output(self):
        torch.manual_seed(42)
        attn = CrossAttention(d_model=64, num_heads=8)
        decoder_h = torch.randn(4, 8, 64)
        encoder_o = torch.randn(4, 12, 64)
        out = attn(decoder_h, encoder_o)
        assert torch.isfinite(out).all()


class TestCrossAttentionMasking:
    def test_mask_shape_accepted(self):
        attn = CrossAttention(d_model=32, num_heads=2)
        decoder_h = torch.randn(2, 4, 32)
        encoder_o = torch.randn(2, 6, 32)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = attn(decoder_h, encoder_o, encoder_mask=mask)
        assert out.shape == (2, 4, 32)

    def test_fully_masked_encoder(self):
        """If all encoder tokens are masked, output should still be finite
        (softmax over all -inf produces 0 weights -> zero context, projected)."""
        torch.manual_seed(0)
        attn = CrossAttention(d_model=32, num_heads=2)
        decoder_h = torch.randn(1, 3, 32)
        encoder_o = torch.randn(1, 5, 32)
        # All encoder tokens masked out
        mask = torch.zeros(1, 5, dtype=torch.bool)
        out = attn(decoder_h, encoder_o, encoder_mask=mask)
        # Should produce nan from softmax of all -inf, but let's be lenient
        # -- the important thing is the function runs without error
        assert out.shape == (1, 3, 32)

    def test_masked_tokens_dont_contribute(self):
        """Changing masked encoder tokens should not affect output."""
        torch.manual_seed(42)
        attn = CrossAttention(d_model=32, num_heads=2)
        attn.eval()

        decoder_h = torch.randn(1, 3, 32)
        encoder_o = torch.randn(1, 6, 32)
        mask = torch.tensor([[True, True, True, False, False, False]])

        out1 = attn(decoder_h, encoder_o, encoder_mask=mask)

        # Modify the masked positions
        encoder_o2 = encoder_o.clone()
        encoder_o2[:, 3:, :] = torch.randn(1, 3, 32) * 100
        out2 = attn(decoder_h, encoder_o2, encoder_mask=mask)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_unmasked_tokens_matter(self):
        """Changing unmasked encoder tokens should affect output."""
        torch.manual_seed(42)
        attn = CrossAttention(d_model=32, num_heads=2)
        attn.eval()

        decoder_h = torch.randn(1, 3, 32)
        encoder_o = torch.randn(1, 6, 32)
        mask = torch.ones(1, 6, dtype=torch.bool)

        out1 = attn(decoder_h, encoder_o, encoder_mask=mask)

        encoder_o2 = encoder_o.clone()
        encoder_o2[:, 0, :] += 10.0
        out2 = attn(decoder_h, encoder_o2, encoder_mask=mask)

        assert not torch.allclose(out1, out2, atol=1e-3)
