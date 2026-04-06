import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

convert_dtype = _mod.convert_dtype
quantize_to_fp8 = _mod.quantize_to_fp8
convert_state_dict = _mod.convert_state_dict


class TestConvertDtype:
    def test_fp32_to_bf16(self):
        w = torch.randn(64, 64, dtype=torch.float32)
        result = convert_dtype(w, torch.bfloat16)
        assert result.dtype == torch.bfloat16
        assert result.shape == w.shape

    def test_fp32_to_fp16(self):
        w = torch.randn(32, 32, dtype=torch.float32)
        result = convert_dtype(w, torch.float16)
        assert result.dtype == torch.float16

    def test_bf16_to_fp32(self):
        w = torch.randn(16, 16, dtype=torch.bfloat16)
        result = convert_dtype(w, torch.float32)
        assert result.dtype == torch.float32

    def test_identity(self):
        w = torch.randn(8, 8, dtype=torch.float32)
        result = convert_dtype(w, torch.float32)
        assert torch.equal(w, result)


class TestQuantizeFP8:
    def test_output_types(self):
        w = torch.randn(32, 32)
        quantized, dequantized, err = quantize_to_fp8(w)
        assert quantized.dtype == torch.float8_e4m3fn
        assert dequantized.dtype == torch.float32
        assert isinstance(err, float)

    def test_shapes_preserved(self):
        w = torch.randn(64, 128)
        quantized, dequantized, _ = quantize_to_fp8(w)
        assert quantized.shape == w.shape
        assert dequantized.shape == w.shape

    def test_error_bounded(self):
        w = torch.randn(100, 100)
        _, dequantized, max_err = quantize_to_fp8(w)
        # Error should be finite and non-negative
        assert max_err >= 0.0
        assert max_err < float("inf")
        # Verify reported error matches actual
        actual_err = (w - dequantized).abs().max().item()
        assert abs(max_err - actual_err) < 1e-7

    def test_zeros(self):
        w = torch.zeros(8, 8)
        quantized, dequantized, max_err = quantize_to_fp8(w)
        assert torch.equal(dequantized, w)
        assert max_err == 0.0

    def test_small_values(self):
        w = torch.tensor([0.001, -0.001, 0.0005])
        _, dequantized, _ = quantize_to_fp8(w)
        # Should approximately preserve small values
        assert torch.allclose(w, dequantized, atol=1e-3)


class TestConvertStateDict:
    def test_basic(self):
        sd = {
            "layer.0.weight": torch.randn(32, 32),
            "layer.1.weight": torch.randn(16, 16),
        }
        converted, errors = convert_state_dict(sd, torch.bfloat16)
        assert len(converted) == 2
        assert len(errors) == 2
        for name in sd:
            assert converted[name].dtype == torch.bfloat16
            assert errors[name] >= 0.0

    def test_fp32_to_fp32_no_error(self):
        sd = {"w": torch.randn(8, 8)}
        converted, errors = convert_state_dict(sd, torch.float32)
        assert errors["w"] == 0.0

    def test_error_values_reasonable(self):
        torch.manual_seed(42)
        sd = {"w": torch.randn(128, 128)}
        _, errors_bf16 = convert_state_dict(sd, torch.bfloat16)
        _, errors_fp16 = convert_state_dict(sd, torch.float16)
        # Both should have small errors for standard normal values
        assert errors_bf16["w"] < 0.1
        assert errors_fp16["w"] < 0.01
