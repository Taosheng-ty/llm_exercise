"""
Tests for Exercise 06: Memory Budget Planner
"""

import importlib.util
import os

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

param_memory_bytes = _mod.param_memory_bytes
optimizer_state_memory_bytes = _mod.optimizer_state_memory_bytes
gradient_memory_bytes = _mod.gradient_memory_bytes
activation_memory_bytes = _mod.activation_memory_bytes
total_training_memory_bytes = _mod.total_training_memory_bytes
max_batch_size = _mod.max_batch_size


class TestParamMemory:
    def test_fp16(self):
        assert param_memory_bytes(1_000_000, 2) == 2_000_000

    def test_fp32(self):
        assert param_memory_bytes(1_000_000, 4) == 4_000_000

    def test_7b_model_fp16(self):
        # 7B params * 2 bytes = 14 GB
        mem = param_memory_bytes(7_000_000_000, 2)
        assert mem == 14_000_000_000


class TestOptimizerStateMemory:
    def test_adam(self):
        # Adam: 2 states * num_params * 4 bytes
        mem = optimizer_state_memory_bytes(1_000_000, "adam", 4)
        assert mem == 8_000_000

    def test_adamw(self):
        mem = optimizer_state_memory_bytes(1_000_000, "adamw", 4)
        assert mem == 8_000_000

    def test_sgd(self):
        mem = optimizer_state_memory_bytes(1_000_000, "sgd", 4)
        assert mem == 0

    def test_sgd_momentum(self):
        mem = optimizer_state_memory_bytes(1_000_000, "sgd_momentum", 4)
        assert mem == 4_000_000


class TestGradientMemory:
    def test_basic(self):
        assert gradient_memory_bytes(1_000_000, 2) == 2_000_000


class TestActivationMemory:
    def test_basic(self):
        mem = activation_memory_bytes(
            batch_size=1, seq_len=4, hidden_dim=32,
            num_layers=1, num_heads=4, ffn_hidden_dim=64, precision_bytes=2,
        )
        # 1 * 4 * (10*32 + 2*4*4 + 4*64) * 2 = 4 * (320 + 32 + 256) * 2 = 4 * 608 * 2 = 4864
        assert mem == 4864

    def test_scales_with_batch(self):
        kwargs = dict(seq_len=8, hidden_dim=32, num_layers=2, num_heads=4,
                      ffn_hidden_dim=64, precision_bytes=2)
        m1 = activation_memory_bytes(batch_size=1, **kwargs)
        m2 = activation_memory_bytes(batch_size=4, **kwargs)
        assert m2 == 4 * m1

    def test_scales_with_layers(self):
        kwargs = dict(batch_size=2, seq_len=8, hidden_dim=32, num_heads=4,
                      ffn_hidden_dim=64, precision_bytes=2)
        m1 = activation_memory_bytes(num_layers=1, **kwargs)
        m2 = activation_memory_bytes(num_layers=3, **kwargs)
        assert m2 == 3 * m1


class TestTotalTrainingMemory:
    def test_sum_of_components(self):
        num_params = 1_000_000
        kwargs = dict(
            batch_size=2, seq_len=8, hidden_dim=32, num_layers=2,
            num_heads=4, ffn_hidden_dim=64, optimizer_type="adam",
            param_precision_bytes=2, optim_precision_bytes=4,
            activation_precision_bytes=2,
        )
        total = total_training_memory_bytes(num_params=num_params, **kwargs)

        p = param_memory_bytes(num_params, 2)
        o = optimizer_state_memory_bytes(num_params, "adam", 4)
        g = gradient_memory_bytes(num_params, 2)
        a = activation_memory_bytes(2, 8, 32, 2, 4, 64, 2)

        assert total == p + o + g + a

    def test_sgd_uses_less_memory(self):
        kwargs = dict(
            num_params=1_000_000, batch_size=2, seq_len=8,
            hidden_dim=32, num_layers=2, num_heads=4, ffn_hidden_dim=64,
            param_precision_bytes=2, optim_precision_bytes=4,
            activation_precision_bytes=2,
        )
        mem_adam = total_training_memory_bytes(optimizer_type="adam", **kwargs)
        mem_sgd = total_training_memory_bytes(optimizer_type="sgd", **kwargs)
        assert mem_sgd < mem_adam


class TestMaxBatchSize:
    def test_basic(self):
        # Give enough memory for a few batches
        num_params = 100_000
        kwargs = dict(
            num_params=num_params, seq_len=8, hidden_dim=32,
            num_layers=2, num_heads=4, ffn_hidden_dim=64,
            optimizer_type="adam", param_precision_bytes=2,
            optim_precision_bytes=4, activation_precision_bytes=2,
        )
        # First figure out memory for batch_size=1
        mem_bs1 = total_training_memory_bytes(batch_size=1, **kwargs)
        # Give 5x that
        gpu_mem = 5 * mem_bs1
        bs = max_batch_size(gpu_memory_bytes=gpu_mem, **kwargs)
        assert bs >= 1

    def test_returns_zero_if_too_small(self):
        bs = max_batch_size(
            gpu_memory_bytes=100,  # way too small
            num_params=1_000_000, seq_len=512, hidden_dim=768,
            num_layers=12, num_heads=12, ffn_hidden_dim=3072,
        )
        assert bs == 0

    def test_larger_gpu_fits_more(self):
        kwargs = dict(
            num_params=100_000, seq_len=8, hidden_dim=32,
            num_layers=2, num_heads=4, ffn_hidden_dim=64,
        )
        bs1 = max_batch_size(gpu_memory_bytes=10_000_000, **kwargs)
        bs2 = max_batch_size(gpu_memory_bytes=100_000_000, **kwargs)
        assert bs2 >= bs1

    def test_consistency_with_total(self):
        """max_batch_size should return a value where total memory fits."""
        kwargs = dict(
            num_params=50_000, seq_len=8, hidden_dim=32,
            num_layers=2, num_heads=4, ffn_hidden_dim=64,
        )
        gpu_mem = 2_000_000
        bs = max_batch_size(gpu_memory_bytes=gpu_mem, **kwargs)
        if bs > 0:
            total = total_training_memory_bytes(batch_size=bs, **kwargs)
            assert total <= gpu_mem
            # And bs+1 should exceed
            total_plus = total_training_memory_bytes(batch_size=bs + 1, **kwargs)
            assert total_plus > gpu_mem
