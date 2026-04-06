"""
Solution for Exercise 04: Convert Model Weights Between dtypes
"""

import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def convert_dtype(
    weight: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a weight tensor to target_dtype."""
    return weight.to(target_dtype)


def quantize_to_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate per-tensor fp8_e4m3fn quantization via scale-and-clamp."""
    weight_fp32 = weight.float()

    # Compute per-tensor scale
    amax = weight_fp32.abs().max()
    scale = amax / FP8_MAX
    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-12)

    # Quantize: scale down, cast to fp8, this does the clamp+round
    scaled = weight_fp32 / scale
    quantized = scaled.to(torch.float8_e4m3fn)

    # Dequantize
    dequantized = quantized.float() * scale

    # Compute max absolute error
    max_abs_error = (weight_fp32 - dequantized).abs().max().item()

    return quantized, dequantized, max_abs_error


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Convert all weights in a state dict to target_dtype."""
    converted = {}
    errors = {}

    for name, tensor in state_dict.items():
        original_fp32 = tensor.float()
        converted_tensor = convert_dtype(tensor, target_dtype)
        converted[name] = converted_tensor

        # Compute max absolute error by comparing in fp32
        back_to_fp32 = converted_tensor.float()
        max_err = (original_fp32 - back_to_fp32).abs().max().item()
        errors[name] = max_err

    return converted, errors
