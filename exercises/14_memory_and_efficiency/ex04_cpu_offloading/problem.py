"""
Exercise 04: CPU Offloading (Medium)

CPU offloading is a technique to reduce GPU memory usage by moving model parameters
to CPU when they are not actively being used. Before a layer's forward pass, its
parameters are moved to GPU ("onloaded"), and after, they are moved back to CPU
("offloaded").

This is used in systems like DeepSpeed ZeRO-Offload, FSDP CPU offload, and
in slime's offload_train/offload_rollout pattern.

Your tasks:
-----------
1. Implement `OffloadWrapper`:
   - __init__(self, module, device="cpu"):
       Wraps an nn.Module. Stores target offload device. Initially offloads
       all parameters to CPU.

   - offload(self):
       Move all parameters (and buffers) of the wrapped module to CPU.
       Return self for chaining.

   - onload(self, device="cuda"):
       Move all parameters (and buffers) of the wrapped module to the specified device.
       If CUDA is not available, use the original device or CPU.
       Return self for chaining.

   - forward(self, *args, **kwargs):
       Automatically onload before forward, run the module, then offload after.
       The input tensors should be on the same device as the module during forward.

2. Implement `offload_forward_pass(layers, input_tensor, device="cpu")`:
   - Given a list of nn.Module layers, run them sequentially.
   - Only one layer's parameters should be on "compute device" at a time.
   - All others should be on CPU.
   - Return the output tensor (on compute device).

Note: For testing without CUDA, we simulate by using different "logical" devices
(actually both CPU, but tracking .device attribute). Tests verify the offload/onload
logic and correctness of the forward pass.
"""

import torch
import torch.nn as nn


class OffloadWrapper(nn.Module):
    """
    Wraps an nn.Module to support parameter offloading to CPU.
    """

    def __init__(self, module: nn.Module, device: str = "cpu"):
        """
        Args:
            module: The nn.Module to wrap.
            device: The device to offload parameters to (default "cpu").
        """
        raise NotImplementedError

    def offload(self) -> "OffloadWrapper":
        """Move all parameters and buffers to the offload device."""
        raise NotImplementedError

    def onload(self, device: str = "cuda") -> "OffloadWrapper":
        """Move all parameters and buffers to the compute device."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Onload, run forward, offload, return output."""
        raise NotImplementedError


def offload_forward_pass(
    layers: list,
    input_tensor: torch.Tensor,
    compute_device: str = "cpu",
) -> torch.Tensor:
    """
    Run layers sequentially, onloading only one at a time.

    Args:
        layers: list of nn.Module layers
        input_tensor: input tensor
        compute_device: device to run computation on

    Returns:
        output tensor on compute_device
    """
    raise NotImplementedError
