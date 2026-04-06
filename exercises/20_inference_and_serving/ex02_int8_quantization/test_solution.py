"""
Tests for Exercise 02: INT8 Quantization
"""

import importlib.util
import os

import pytest
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

quantize_symmetric = _mod.quantize_symmetric
dequantize_symmetric = _mod.dequantize_symmetric
quantize_asymmetric = _mod.quantize_asymmetric
dequantize_asymmetric = _mod.dequantize_asymmetric
compute_quantization_error = _mod.compute_quantization_error


class TestSymmetricQuantization:
    def test_output_dtype_is_int8(self):
        t = torch.randn(4, 8)
        q, scale = quantize_symmetric(t)
        assert q.dtype == torch.int8

    def test_scale_shape_per_channel(self):
        t = torch.randn(4, 8)
        q, scale = quantize_symmetric(t)
        assert scale.shape[0] == 4
        assert scale.numel() == 4  # one scale per channel

    def test_quantized_values_in_range(self):
        t = torch.randn(8, 16) * 10
        q, scale = quantize_symmetric(t)
        assert q.min().item() >= -127
        assert q.max().item() <= 127

    def test_roundtrip_error_is_small(self):
        torch.manual_seed(42)
        t = torch.randn(4, 32)
        q, scale = quantize_symmetric(t)
        reconstructed = dequantize_symmetric(q, scale)
        # INT8 symmetric should have small relative error
        max_err = (t - reconstructed).abs().max().item()
        # Error bounded by 1 quantization step per channel
        assert max_err < t.abs().max().item() / 50  # loose bound

    def test_zero_tensor(self):
        t = torch.zeros(2, 4)
        q, scale = quantize_symmetric(t)
        assert (q == 0).all()


class TestAsymmetricQuantization:
    def test_output_dtype_is_uint8(self):
        t = torch.randn(4, 8)
        q, scale, zp = quantize_asymmetric(t)
        assert q.dtype == torch.uint8

    def test_quantized_values_in_range(self):
        t = torch.randn(8, 16) * 5
        q, scale, zp = quantize_asymmetric(t)
        assert q.min().item() >= 0
        assert q.max().item() <= 255

    def test_roundtrip_error_is_small(self):
        torch.manual_seed(42)
        t = torch.randn(4, 32) * 3
        q, scale, zp = quantize_asymmetric(t)
        reconstructed = dequantize_asymmetric(q, scale, zp)
        max_err = (t - reconstructed).abs().max().item()
        assert max_err < t.abs().max().item() / 50

    def test_positive_only_tensor(self):
        """Asymmetric should handle non-negative tensors well."""
        t = torch.rand(2, 8) * 10  # all positive
        q, scale, zp = quantize_asymmetric(t)
        reconstructed = dequantize_asymmetric(q, scale, zp)
        assert (t - reconstructed).abs().max().item() < 1.0


class TestQuantizationError:
    def test_error_keys(self):
        t = torch.randn(4, 8)
        q, scale = quantize_symmetric(t)
        recon = dequantize_symmetric(q, scale)
        err = compute_quantization_error(t, recon)
        assert "mse" in err
        assert "max_abs_error" in err
        assert "snr_db" in err

    def test_perfect_reconstruction_high_snr(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        err = compute_quantization_error(t, t)
        assert err["mse"] < 1e-10
        assert err["max_abs_error"] < 1e-10

    def test_snr_positive_for_good_quantization(self):
        torch.manual_seed(42)
        t = torch.randn(16, 64)
        q, scale = quantize_symmetric(t)
        recon = dequantize_symmetric(q, scale)
        err = compute_quantization_error(t, recon)
        # INT8 should give > 30 dB SNR for Gaussian data
        assert err["snr_db"] > 30.0

    def test_mse_decreases_with_more_bits_concept(self):
        """Symmetric quantization error should be reasonable for 8-bit."""
        torch.manual_seed(0)
        t = torch.randn(8, 128)
        q, scale = quantize_symmetric(t, num_bits=8)
        recon = dequantize_symmetric(q, scale)
        err = compute_quantization_error(t, recon)
        # MSE should be small relative to signal variance
        assert err["mse"] < t.var().item() * 0.01
