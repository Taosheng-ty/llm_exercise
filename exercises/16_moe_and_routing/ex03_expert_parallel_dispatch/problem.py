"""
Exercise 03: Expert Parallel Token Dispatch (Hard, PyTorch)

Simulate the token dispatch mechanism used in expert parallelism for MoE models.
In real distributed training, tokens are scattered to the GPU holding each expert,
processed, and gathered back. Here we simulate this on a single device.

Your task:
    Implement `expert_dispatch(hidden_states, routing_weights, routing_indices,
                               expert_ffns, num_experts)` that:
    1. For each expert, collects all tokens assigned to it
    2. Processes those tokens through the corresponding expert FFN
    3. Combines outputs weighted by routing_weights
    4. Returns the final output in original token order

    Each expert FFN is a callable: expert_ffns[i](x) -> y, where x and y have the
    same last dimension.
"""

import torch


def expert_dispatch(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    routing_indices: torch.Tensor,
    expert_ffns: list,
    num_experts: int,
) -> torch.Tensor:
    """
    Args:
        hidden_states: (num_tokens, dim) - input token representations
        routing_weights: (num_tokens, top_k) - softmax weights for selected experts
        routing_indices: (num_tokens, top_k) - expert indices per token
        expert_ffns: list of num_experts callables, each (batch, dim) -> (batch, dim)
        num_experts: total number of experts

    Returns:
        output: (num_tokens, dim) - combined expert outputs
    """
    raise NotImplementedError("Implement expert parallel dispatch")
