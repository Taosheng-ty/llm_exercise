"""
Exercise 01: Top-K Expert Routing (Medium, PyTorch)

Implement the core routing mechanism used in Mixture of Experts (MoE) models.
Given hidden states and a router weight matrix, compute which experts each token
should be sent to and what weight each expert receives.

Top-k routing is the core mechanism in Mixture-of-Experts LLMs (Mixtral,
DeepSeek-MoE) that enables scaling model parameters without proportionally
scaling compute. Each token is processed by only k of N experts, allowing models
with hundreds of billions of parameters to run at the cost of a much smaller
dense model.

Inspired by MoE routing in slime's Megatron backend for models like Qwen3-MoE
and DeepSeek-V3.

Your task:
    Implement `top_k_routing(hidden_states, router_weights, top_k)` that:
    1. Computes router logits = hidden_states @ router_weights  (shape: [num_tokens, num_experts])
    2. Selects the top-k experts per token
    3. Applies softmax ONLY over the selected top-k logits (not all experts)
    4. Returns:
       - routing_weights: (num_tokens, top_k) float tensor of softmax weights
       - routing_indices: (num_tokens, top_k) long tensor of expert indices
"""

import torch


def top_k_routing(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states: (num_tokens, dim) - token representations
        router_weights: (dim, num_experts) - router projection matrix
        top_k: number of experts to select per token

    Returns:
        routing_weights: (num_tokens, top_k) - softmax weights for selected experts
        routing_indices: (num_tokens, top_k) - indices of selected experts
    """
    raise NotImplementedError("Implement top-k expert routing")
