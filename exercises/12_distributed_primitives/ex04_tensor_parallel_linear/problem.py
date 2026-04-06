"""
Exercise 04: Tensor Parallel Linear Layer (Hard, PyTorch)

Simulate tensor parallelism for a linear layer. In Megatron-LM style tensor
parallelism, a single linear layer's weight is split across multiple GPUs.

There are two flavors:

1. Column Parallel: Weight W (out_features x in_features) is split along the
   output dimension. Each shard computes a slice of the output. The full output
   is obtained by all-gather (concatenation along the output dim).

2. Row Parallel: Weight W is split along the input dimension. The input X is
   split accordingly. Each shard computes a partial output. The full output is
   obtained by all-reduce (element-wise sum).

Implement two functions:

    column_parallel_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        num_shards: int,
    ) -> torch.Tensor

    row_parallel_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        num_shards: int,
    ) -> torch.Tensor

Both should:
    1. Split the weight (and bias where applicable) into num_shards pieces.
    2. Simulate each shard computing its portion.
    3. Combine results (all-gather for column, all-reduce for row).
    4. Return the same result as a standard nn.Linear with the given weight/bias.

Bias handling:
    - Column parallel: bias is split along the output dimension (each shard gets
      its slice of the bias), then concatenated after all-gather.
    - Row parallel: bias is NOT split. Each shard computes a partial output
      (without bias), the partial outputs are summed (all-reduce), and then
      the full bias is added once to the combined result.

Args:
    input: Tensor of shape (batch_size, in_features).
    weight: Tensor of shape (out_features, in_features) -- same as nn.Linear.weight.
    bias: Optional tensor of shape (out_features,) or None.
    num_shards: Number of tensor parallel shards.

Returns:
    Output tensor of shape (batch_size, out_features).
"""

import torch


def column_parallel_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_shards: int,
) -> torch.Tensor:
    """
    Simulate column-parallel linear: split weight along output dim.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement column_parallel_linear")


def row_parallel_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    num_shards: int,
) -> torch.Tensor:
    """
    Simulate row-parallel linear: split weight along input dim.

    TODO: Implement this function.
    """
    raise NotImplementedError("Implement row_parallel_linear")
