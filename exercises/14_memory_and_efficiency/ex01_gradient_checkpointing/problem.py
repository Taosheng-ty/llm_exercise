"""
Exercise 01: Gradient Checkpointing (Hard)

Gradient checkpointing (also called activation checkpointing) is a technique to reduce
peak GPU memory usage during training. Instead of storing all intermediate activations
for the backward pass, we only store activations at "checkpoint" boundaries and recompute
the rest during backward.

This trades compute for memory -- you do ~33% more compute but can train much larger
models or use larger batch sizes.

Your tasks:
-----------
1. Implement `checkpoint_sequential(functions, input_tensor)`:
   - Takes a list of callables (layers) and an input tensor.
   - Runs them sequentially but uses `torch.utils.checkpoint.checkpoint` to wrap
     groups of layers (each layer is individually checkpointed).
   - Returns the final output tensor.

2. Implement `ManualCheckpointFunction` (a custom torch.autograd.Function):
   - `forward(ctx, input_tensor, *functions)`:
       * Run all functions sequentially on input_tensor.
       * Save only the input_tensor and the functions in ctx (NOT intermediate activations).
       * Return the final output.
   - `backward(ctx, grad_output)`:
       * IMPORTANT: Wrap recomputation in `torch.enable_grad()` context, because
         autograd is disabled by default inside custom autograd backward functions.
       * Recompute forward pass from saved input to get intermediates.
       * Compute gradients by backpropagating through recomputed graph.
       * Return grad w.r.t. input_tensor (and None for each function).

3. Implement `manual_checkpoint_sequential(functions, input_tensor)`:
   - Uses ManualCheckpointFunction.apply to run the functions with manual checkpointing.

All functions should produce the same numerical output as a plain sequential forward pass.
"""

import torch
import torch.nn as nn


def checkpoint_sequential(functions: list, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run functions sequentially with torch.utils.checkpoint.checkpoint on each.

    Args:
        functions: list of callables (e.g., nn.Module layers)
        input_tensor: the input tensor (requires_grad=True)

    Returns:
        output tensor after applying all functions in sequence
    """
    raise NotImplementedError("Implement checkpoint_sequential")


class ManualCheckpointFunction(torch.autograd.Function):
    """
    Custom autograd function that saves only input and recomputes
    intermediate activations during backward.
    """

    @staticmethod
    def forward(ctx, input_tensor, *functions):
        """
        Run functions sequentially. Save only input_tensor and functions.
        Do NOT save intermediate activations.
        """
        raise NotImplementedError("Implement ManualCheckpointFunction.forward")

    @staticmethod
    def backward(ctx, grad_output):
        """
        Recompute forward from saved input, then backprop through the
        recomputed graph to get gradients.

        Note: You must use torch.enable_grad() as a context manager here,
        because PyTorch disables autograd inside custom backward functions.
        """
        raise NotImplementedError("Implement ManualCheckpointFunction.backward")


def manual_checkpoint_sequential(
    functions: list, input_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Run functions sequentially using ManualCheckpointFunction.

    Args:
        functions: list of callables
        input_tensor: input tensor (requires_grad=True)

    Returns:
        output tensor
    """
    raise NotImplementedError("Implement manual_checkpoint_sequential")
