"""
Tests for Exercise 03: Activation Memory Estimation
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

estimate_attention_activation_memory = _mod.estimate_attention_activation_memory
estimate_ffn_activation_memory = _mod.estimate_ffn_activation_memory
estimate_transformer_layer_memory = _mod.estimate_transformer_layer_memory
estimate_total_activation_memory = _mod.estimate_total_activation_memory
measure_actual_activation_memory = _mod.measure_actual_activation_memory


class TestAttentionMemory:
    def test_basic(self):
        # B=2, H=4, S=8, D=16, fp16 (2 bytes)
        mem = estimate_attention_activation_memory(2, 4, 8, 16, dtype_bytes=2)
        # Q,K,V: 3 * 2*4*8*16*2 = 6144
        # attn scores + softmax: 2 * 2*4*8*8*2 = 2048
        # attn output: 2*4*8*16*2 = 2048
        expected = 6144 + 2048 + 2048
        assert mem == expected, f"Expected {expected}, got {mem}"

    def test_scales_with_batch(self):
        mem1 = estimate_attention_activation_memory(1, 4, 8, 16, dtype_bytes=2)
        mem2 = estimate_attention_activation_memory(2, 4, 8, 16, dtype_bytes=2)
        assert mem2 == 2 * mem1

    def test_scales_with_seq_len_quadratic_in_scores(self):
        """Attention scores scale quadratically with seq_len."""
        mem1 = estimate_attention_activation_memory(1, 1, 4, 8, dtype_bytes=2)
        mem2 = estimate_attention_activation_memory(1, 1, 8, 8, dtype_bytes=2)
        # Not exactly 4x because QKV and output scale linearly
        assert mem2 > mem1

    def test_fp32(self):
        mem_fp16 = estimate_attention_activation_memory(1, 2, 4, 8, dtype_bytes=2)
        mem_fp32 = estimate_attention_activation_memory(1, 2, 4, 8, dtype_bytes=4)
        assert mem_fp32 == 2 * mem_fp16


class TestFFNMemory:
    def test_basic(self):
        # B=2, S=8, hidden=32, ffn=64, fp16
        mem = estimate_ffn_activation_memory(2, 8, 32, 64, dtype_bytes=2)
        # 4 tensors of [2, 8, 64] * 2 bytes = 4 * 2*8*64*2 = 8192
        expected = 4 * 2 * 8 * 64 * 2
        assert mem == expected, f"Expected {expected}, got {mem}"

    def test_scales_with_batch(self):
        mem1 = estimate_ffn_activation_memory(1, 8, 32, 64, dtype_bytes=2)
        mem2 = estimate_ffn_activation_memory(4, 8, 32, 64, dtype_bytes=2)
        assert mem2 == 4 * mem1


class TestTransformerLayerMemory:
    def test_includes_attention_ffn_residual(self):
        B, S, H, NH, FFN = 2, 8, 32, 4, 64
        head_dim = H // NH
        attn = estimate_attention_activation_memory(B, NH, S, head_dim, 2)
        ffn = estimate_ffn_activation_memory(B, S, H, FFN, 2)
        residual = 2 * B * S * H * 2
        layer = estimate_transformer_layer_memory(B, S, H, NH, FFN, 2)
        assert layer == attn + ffn + residual

    def test_positive(self):
        mem = estimate_transformer_layer_memory(1, 16, 64, 4, 128, 2)
        assert mem > 0


class TestTotalActivationMemory:
    def test_scales_with_layers(self):
        kwargs = dict(batch_size=1, seq_len=8, hidden_dim=32, num_heads=4, ffn_hidden_dim=64, dtype_bytes=2)
        mem1 = estimate_total_activation_memory(num_layers=1, **kwargs)
        mem2 = estimate_total_activation_memory(num_layers=2, **kwargs)
        # Difference should be exactly one layer's worth
        layer_mem = estimate_transformer_layer_memory(1, 8, 32, 4, 64, 2)
        assert mem2 - mem1 == layer_mem

    def test_includes_embedding(self):
        kwargs = dict(batch_size=2, seq_len=8, hidden_dim=32, num_heads=4, ffn_hidden_dim=64, dtype_bytes=2)
        total = estimate_total_activation_memory(num_layers=0, **kwargs)
        # With 0 layers, should just be embedding
        embedding = 2 * 8 * 32 * 2
        assert total == embedding


class TestMeasureActualMemory:
    def test_simple_model(self):
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        x = torch.randn(4, 16)
        mem = measure_actual_activation_memory(model, x)
        # Should be positive
        assert mem > 0

    def test_larger_model_uses_more_memory(self):
        small = nn.Sequential(nn.Linear(16, 16))
        large = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16))
        x = torch.randn(4, 16)
        mem_small = measure_actual_activation_memory(small, x)
        mem_large = measure_actual_activation_memory(large, x)
        assert mem_large > mem_small

    def test_scales_with_batch(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        x1 = torch.randn(2, 8)
        x2 = torch.randn(4, 8)
        mem1 = measure_actual_activation_memory(model, x1)
        mem2 = measure_actual_activation_memory(model, x2)
        assert mem2 == 2 * mem1
