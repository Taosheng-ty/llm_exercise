"""
Solution for Exercise 04: CPU Offloading
"""

import torch
import torch.nn as nn


class OffloadWrapper(nn.Module):
    """
    Wraps an nn.Module to support parameter offloading to CPU.
    """

    def __init__(self, module: nn.Module, device: str = "cpu"):
        super().__init__()
        self.module = module
        self.offload_device = device
        # Initially offload to CPU
        self.offload()

    def offload(self) -> "OffloadWrapper":
        """Move all parameters and buffers to the offload device."""
        self.module.to(self.offload_device)
        return self

    def onload(self, device: str = "cuda") -> "OffloadWrapper":
        """Move all parameters and buffers to the compute device."""
        # If CUDA requested but not available, fall back to CPU
        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
        self.module.to(device)
        return self

    def forward(self, *args, **kwargs):
        """Onload, run forward, offload, return output."""
        # Determine compute device from input
        compute_device = "cpu"
        for a in args:
            if isinstance(a, torch.Tensor):
                compute_device = str(a.device)
                break

        self.onload(compute_device)
        output = self.module(*args, **kwargs)
        self.offload()
        return output


def offload_forward_pass(
    layers: list,
    input_tensor: torch.Tensor,
    compute_device: str = "cpu",
) -> torch.Tensor:
    """
    Run layers sequentially, onloading only one at a time.
    """
    output = input_tensor.to(compute_device)

    for layer in layers:
        # Onload this layer
        layer.to(compute_device)
        # Forward
        output = layer(output)
        # Offload back to CPU
        layer.to("cpu")

    return output
