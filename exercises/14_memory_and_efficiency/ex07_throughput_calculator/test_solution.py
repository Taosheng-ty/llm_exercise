"""
Tests for Exercise 07: Throughput Calculator
"""

import importlib.util
import os
import math

import numpy as np

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

tokens_per_second = _mod.tokens_per_second
samples_per_second = _mod.samples_per_second
achieved_tflops = _mod.achieved_tflops
model_flops_utilization = _mod.model_flops_utilization
compute_all_metrics = _mod.compute_all_metrics


class TestTokensPerSecond:
    def test_basic(self):
        # 4 * 512 * 8 / 2.0 = 8192 tokens/sec
        result = tokens_per_second(4, 512, 8, 2.0)
        assert abs(result - 8192.0) < 1e-6

    def test_single_gpu(self):
        result = tokens_per_second(1, 100, 1, 1.0)
        assert abs(result - 100.0) < 1e-6

    def test_scales_with_batch(self):
        t1 = tokens_per_second(1, 100, 1, 1.0)
        t2 = tokens_per_second(4, 100, 1, 1.0)
        assert abs(t2 - 4 * t1) < 1e-6

    def test_scales_with_gpus(self):
        t1 = tokens_per_second(4, 100, 1, 1.0)
        t2 = tokens_per_second(4, 100, 8, 1.0)
        assert abs(t2 - 8 * t1) < 1e-6

    def test_inversely_proportional_to_time(self):
        t1 = tokens_per_second(4, 100, 1, 1.0)
        t2 = tokens_per_second(4, 100, 1, 2.0)
        assert abs(t1 - 2 * t2) < 1e-6


class TestSamplesPerSecond:
    def test_basic(self):
        result = samples_per_second(4, 8, 2.0)
        assert abs(result - 16.0) < 1e-6

    def test_single(self):
        result = samples_per_second(1, 1, 1.0)
        assert abs(result - 1.0) < 1e-6


class TestAchievedTFLOPs:
    def test_basic(self):
        # flops_per_token=1e6, bs=4, seq=512, gpus=1, time=1.0
        # total_flops = 1e6 * 4 * 512 * 1 = 2.048e9
        # training = 3 * 2.048e9 = 6.144e9
        # tflops = 6.144e9 / 1e12 = 0.006144
        # tflops/s = 0.006144 / 1.0 = 0.006144
        result = achieved_tflops(1e6, 4, 512, 1, 1.0)
        assert abs(result - 0.006144) < 1e-8

    def test_scales_with_gpus(self):
        t1 = achieved_tflops(1e6, 4, 512, 1, 1.0)
        t2 = achieved_tflops(1e6, 4, 512, 8, 1.0)
        assert abs(t2 - 8 * t1) < 1e-8

    def test_is_3x_forward(self):
        """Training TFLOPs should be 3x forward-only TFLOPs."""
        flops_per_token = 1e6
        total_fwd_flops = flops_per_token * 4 * 512 * 1
        fwd_tflops_per_sec = total_fwd_flops / 1e12 / 1.0
        training_tflops = achieved_tflops(1e6, 4, 512, 1, 1.0)
        assert abs(training_tflops - 3 * fwd_tflops_per_sec) < 1e-10


class TestMFU:
    def test_basic(self):
        # 100 TFLOPs achieved, peak=312 per GPU, 1 GPU
        mfu = model_flops_utilization(100.0, 312.0, 1)
        assert abs(mfu - 100.0 / 312.0) < 1e-6

    def test_multi_gpu(self):
        mfu = model_flops_utilization(500.0, 312.0, 8)
        assert abs(mfu - 500.0 / (312.0 * 8)) < 1e-6

    def test_perfect_utilization(self):
        mfu = model_flops_utilization(312.0, 312.0, 1)
        assert abs(mfu - 1.0) < 1e-6

    def test_between_0_and_1_typical(self):
        mfu = model_flops_utilization(150.0, 312.0, 1)
        assert 0.0 < mfu < 1.0


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        result = compute_all_metrics(
            batch_size=4, seq_len=512, num_gpus=8,
            step_time_seconds=2.0, model_flops_per_token=1e6,
            peak_tflops_per_gpu=312.0,
        )
        assert "tokens_per_sec" in result
        assert "samples_per_sec" in result
        assert "tflops" in result
        assert "mfu" in result

    def test_values_consistent(self):
        kwargs = dict(
            batch_size=4, seq_len=512, num_gpus=8,
            step_time_seconds=2.0, model_flops_per_token=1e6,
            peak_tflops_per_gpu=312.0,
        )
        result = compute_all_metrics(**kwargs)

        # Check individual functions match
        assert abs(result["tokens_per_sec"] - tokens_per_second(4, 512, 8, 2.0)) < 1e-6
        assert abs(result["samples_per_sec"] - samples_per_second(4, 8, 2.0)) < 1e-6
        tflops_val = achieved_tflops(1e6, 4, 512, 8, 2.0)
        assert abs(result["tflops"] - tflops_val) < 1e-10
        mfu_val = model_flops_utilization(tflops_val, 312.0, 8)
        assert abs(result["mfu"] - mfu_val) < 1e-10

    def test_mfu_reasonable_range(self):
        """For a realistic scenario, MFU should be between 0 and 1."""
        result = compute_all_metrics(
            batch_size=32, seq_len=2048, num_gpus=8,
            step_time_seconds=1.5, model_flops_per_token=1e8,
            peak_tflops_per_gpu=312.0,
        )
        assert 0.0 < result["mfu"]  # Should be positive

    def test_all_positive(self):
        result = compute_all_metrics(
            batch_size=1, seq_len=128, num_gpus=1,
            step_time_seconds=0.5, model_flops_per_token=1e5,
            peak_tflops_per_gpu=312.0,
        )
        for key, val in result.items():
            assert val > 0, f"{key} should be positive, got {val}"
