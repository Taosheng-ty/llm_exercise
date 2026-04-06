"""
Solution for Exercise 01: Gradient Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def checkpoint_sequential(functions: list, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run functions sequentially with torch.utils.checkpoint.checkpoint on each.
    """
    output = input_tensor
    for fn in functions:
        # use_reentrant=False is the recommended approach in modern PyTorch
        output = checkpoint(fn, output, use_reentrant=False)
    return output


class ManualCheckpointFunction(torch.autograd.Function):
    """
    Custom autograd function that saves only input and recomputes
    intermediate activations during backward.
    """

    @staticmethod
    def forward(ctx, input_tensor, *functions):
        # Save functions for backward
        ctx.functions = functions
        # Save input tensor for recomputation
        ctx.save_for_backward(input_tensor)

        # Run forward WITHOUT tracking gradients for intermediates
        # (this is the memory saving -- we don't store intermediate activations)
        with torch.no_grad():
            output = input_tensor
            for fn in functions:
                output = fn(output)
        # We need to return a tensor that is connected to the graph
        # detach and re-require grad so autograd can flow through
        output = output.detach()
        output.requires_grad_(input_tensor.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        functions = ctx.functions

        # Recompute forward WITH gradient tracking
        input_tensor = input_tensor.detach().requires_grad_(True)
        with torch.enable_grad():
            output = input_tensor
            for fn in functions:
                output = fn(output)

        # Backpropagate through the recomputed graph
        torch.autograd.backward(output, grad_output)

        # Return gradients: one for input_tensor, None for each function
        return (input_tensor.grad,) + (None,) * len(functions)


def manual_checkpoint_sequential(
    functions: list, input_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Run functions sequentially using ManualCheckpointFunction.
    """
    return ManualCheckpointFunction.apply(input_tensor, *functions)
