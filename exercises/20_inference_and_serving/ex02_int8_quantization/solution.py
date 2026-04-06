"""
Solution for Exercise 02: INT8 Quantization
"""

import torch


def quantize_symmetric(tensor: torch.Tensor, num_bits: int = 8):
    """
    Per-channel symmetric quantization along dim 0.

    Returns:
        (quantized_int8, scale) where scale has shape (num_channels, 1, ...)
    """
    qmax = 2 ** (num_bits - 1) - 1  # 127 for 8 bits

    # Compute per-channel max absolute value along all dims except dim 0
    reduce_dims = list(range(1, tensor.ndim))
    if reduce_dims:
        channel_max = tensor.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        channel_max = tensor.abs().unsqueeze(-1)

    # Avoid division by zero
    channel_max = torch.clamp(channel_max, min=1e-8)

    scale = channel_max / qmax  # shape: (C, 1, ...)

    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)

    return quantized, scale


def dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize symmetrically quantized tensor."""
    return quantized.float() * scale


def quantize_asymmetric(tensor: torch.Tensor, num_bits: int = 8):
    """
    Per-channel asymmetric quantization along dim 0.

    Returns:
        (quantized_uint8, scale, zero_point)
    """
    qmax = 2**num_bits - 1  # 255 for 8 bits

    reduce_dims = list(range(1, tensor.ndim))
    if reduce_dims:
        channel_min = tensor.amin(dim=reduce_dims, keepdim=True)
        channel_max = tensor.amax(dim=reduce_dims, keepdim=True)
    else:
        channel_min = tensor.unsqueeze(-1)
        channel_max = tensor.unsqueeze(-1)

    # Avoid zero range
    range_val = channel_max - channel_min
    range_val = torch.clamp(range_val, min=1e-8)

    scale = range_val / qmax
    zero_point = torch.round(-channel_min / scale).clamp(0, qmax)

    quantized = torch.round(tensor / scale + zero_point).clamp(0, qmax).to(torch.uint8)

    return quantized, scale, zero_point


def dequantize_asymmetric(
    quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize asymmetrically quantized tensor."""
    return (quantized.float() - zero_point.float()) * scale


def compute_quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """
    Compute quantization error metrics.

    Returns:
        dict with keys "mse", "max_abs_error", "snr_db"
    """
    diff = original - reconstructed
    mse = (diff**2).mean().item()
    max_abs_error = diff.abs().max().item()

    signal_power = (original**2).mean().item()
    # Avoid log of zero
    if mse < 1e-20:
        snr_db = float("inf")
    else:
        snr_db = 10.0 * torch.log10(torch.tensor(signal_power / mse)).item()

    return {"mse": mse, "max_abs_error": max_abs_error, "snr_db": snr_db}
