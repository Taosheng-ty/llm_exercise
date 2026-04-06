"""
Exercise 04: Convert Model Weights Between dtypes (Easy-Medium, PyTorch)

When deploying models, we often need to convert weights to lower precision
for memory and speed. Common conversions:
  - fp32 -> bf16  (training default to inference)
  - fp32 -> fp16  (training to deployment)
  - bf16 -> fp8_e4m3  (quantization for fast inference)

For fp8 simulation, we clamp values to the fp8 range, compute a per-tensor
scale, and quantize/dequantize (since not all hardware supports native fp8).

Reference: slime/tools/convert_hf_to_fp8.py

Tasks:
    1. Implement convert_dtype() for fp32->bf16 and fp32->fp16.
    2. Implement quantize_to_fp8() that simulates per-tensor fp8_e4m3 quantization.
    3. Implement convert_state_dict() to convert an entire state dict, tracking max error.
"""

import torch


def convert_dtype(
    weight: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a weight tensor to target_dtype.

    Args:
        weight: input tensor (any dtype)
        target_dtype: one of torch.bfloat16, torch.float16, torch.float32

    Returns:
        Weight tensor in the target dtype.
    """
    # TODO: Implement this function
    raise NotImplementedError


def quantize_to_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Simulate per-tensor fp8_e4m3fn quantization via scale-and-clamp.

    Steps:
      1. Compute scale = max(|weight|) / fp8_max  (where fp8_max = 448.0 for e4m3fn)
      2. Quantized = clamp(weight / scale, -fp8_max, fp8_max), rounded to nearest integer
         representation (use .to(torch.float8_e4m3fn))
      3. Dequantized = quantized.float() * scale

    Args:
        weight: input tensor in fp32 or bf16

    Returns:
        (quantized_fp8, dequantized_fp32, max_abs_error)
        - quantized_fp8: the fp8 tensor
        - dequantized_fp32: the dequantized tensor back in float32
        - max_abs_error: maximum absolute error between original and dequantized
    """
    # TODO: Implement this function
    raise NotImplementedError


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Convert all weights in a state dict to target_dtype.

    Args:
        state_dict: mapping of name -> tensor
        target_dtype: target dtype (torch.bfloat16, torch.float16, or torch.float32)

    Returns:
        (converted_state_dict, error_dict) where error_dict maps each name
        to the max absolute error introduced by the conversion.
    """
    # TODO: Implement this function
    raise NotImplementedError
