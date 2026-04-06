"""
Exercise 05: Shared + Routed Expert Architecture (Medium, PyTorch)

Implement the shared expert pattern used in models like Qwen3-MoE and DeepSeek-V2.
In this architecture, each token is processed through:
  1. A shared expert (always active for every token)
  2. Top-k routed experts (selected per-token by the router)

The final output combines both:
    output = shared_expert(x) + sum_k(routing_weight_k * expert_k(x))

The shared expert architecture (used in DeepSeek-MoE) ensures that common
knowledge is handled by a dedicated expert available to all tokens, while routed
experts specialize in domain-specific patterns. This improves training stability
and prevents the "capacity waste" problem where multiple routed experts learn
redundant representations.

Reference: Qwen3-MoE shared expert pattern in slime's model configurations.

Your task:
    Implement `SharedMoELayer` with a forward method.

    Required attribute names (accessed by tests):
    - self.router: nn.Linear router layer (dim -> num_routed_experts)
    - self.shared_expert: the shared expert module (always-active)
    - self.routed_experts: nn.ModuleList of routed expert modules
"""

import torch
import torch.nn as nn


class SharedMoELayer(nn.Module):
    """
    MoE layer with a shared expert + top-k routed experts.

    Args:
        dim: hidden dimension
        num_routed_experts: number of routed (sparse) experts
        top_k: number of routed experts activated per token
        intermediate_dim: FFN intermediate dimension (default: 4 * dim)
    """

    def __init__(self, dim: int, num_routed_experts: int, top_k: int, intermediate_dim: int = None):
        super().__init__()
        raise NotImplementedError("Implement SharedMoELayer.__init__")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (num_tokens, dim)

        Returns:
            output: (num_tokens, dim)
        """
        raise NotImplementedError("Implement SharedMoELayer.forward")
