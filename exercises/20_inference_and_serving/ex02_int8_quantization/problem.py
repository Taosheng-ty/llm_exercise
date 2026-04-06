"""
Exercise 02: INT8 Quantization (Medium)

Quantization reduces model size and speeds up inference by representing weights
and activations with fewer bits. INT8 quantization maps floating-point values to
8-bit integers using a scale (and optionally a zero-point).

Two main schemes:
- Symmetric: zero maps to 0, range is [-127, 127]. Only needs a scale factor.
- Asymmetric: maps [min, max] to [0, 255]. Needs both scale and zero_point.

Per-channel quantization computes separate scale/zero_point for each output channel,
giving better accuracy than per-tensor (single scale for the whole tensor).

Your tasks:
-----------
1. Implement `quantize_symmetric(tensor, num_bits=8) -> (quantized, scale)`:
   - Per-channel along dim 0: for each row, scale = max(|row|) / (2^(num_bits-1) - 1)
   - quantized = round(tensor / scale), clamped to [-127, 127]
   - Return (quantized as int8, scale as float tensor)

2. Implement `dequantize_symmetric(quantized, scale) -> tensor`:
   - Reconstruct: tensor = quantized.float() * scale

3. Implement `quantize_asymmetric(tensor, num_bits=8) -> (quantized, scale, zero_point)`:
   - Per-channel along dim 0
   - scale = (max - min) / (2^num_bits - 1)
   - zero_point = round(-min / scale), clamped to [0, 255]
   - quantized = round(tensor / scale) + zero_point, clamped to [0, 255]
   - Return (quantized as uint8, scale, zero_point)

4. Implement `dequantize_asymmetric(quantized, scale, zero_point) -> tensor`:
   - Reconstruct: tensor = (quantized.float() - zero_point.float()) * scale

5. Implement `compute_quantization_error(original, reconstructed) -> dict`:
   - Returns dict with keys: "mse", "max_abs_error", "snr_db"
   - MSE = mean squared error
   - max_abs_error = max |original - reconstructed|
   - SNR (dB) = 10 * log10(signal_power / noise_power)
     where signal_power = mean(original^2), noise_power = MSE
"""

import torch


def quantize_symmetric(tensor: torch.Tensor, num_bits: int = 8):
    """
    Per-channel symmetric quantization along dim 0.

    Returns:
        (quantized_int8, scale) where scale has shape (num_channels, 1, ...)
    """
    raise NotImplementedError("Implement quantize_symmetric")


def dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize symmetrically quantized tensor."""
    raise NotImplementedError("Implement dequantize_symmetric")


def quantize_asymmetric(tensor: torch.Tensor, num_bits: int = 8):
    """
    Per-channel asymmetric quantization along dim 0.

    Returns:
        (quantized_uint8, scale, zero_point)
    """
    raise NotImplementedError("Implement quantize_asymmetric")


def dequantize_asymmetric(
    quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize asymmetrically quantized tensor."""
    raise NotImplementedError("Implement dequantize_asymmetric")


def compute_quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """
    Compute quantization error metrics.

    Returns:
        dict with keys "mse", "max_abs_error", "snr_db"
    """
    raise NotImplementedError("Implement compute_quantization_error")
