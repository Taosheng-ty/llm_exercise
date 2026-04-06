import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

load_balancing_loss = _mod.load_balancing_loss


def test_returns_scalar():
    logits = torch.randn(10, 4)
    indices = torch.randint(0, 4, (10, 2))
    loss = load_balancing_loss(logits, indices, num_experts=4)
    assert loss.dim() == 0, "Loss should be a scalar"


def test_uniform_routing_minimal_loss():
    """Perfectly uniform routing should yield loss close to 1.0 (the theoretical minimum)."""
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 100, 4, 1

    # Uniform router logits -> uniform P_i = 1/num_experts
    router_logits = torch.zeros(num_tokens, num_experts)

    # Perfectly balanced routing: each expert gets exactly num_tokens/num_experts tokens
    indices = torch.arange(num_tokens).remainder(num_experts).unsqueeze(1)

    loss = load_balancing_loss(router_logits, indices, num_experts)

    # With uniform f and P, loss = num_experts * sum(1/E * 1/E) = num_experts * E * (1/E^2) = 1
    assert torch.isclose(loss, torch.tensor(1.0), atol=0.05), f"Uniform loss should be ~1.0, got {loss.item()}"


def test_skewed_routing_high_loss():
    """All tokens routed to a single expert should yield high loss."""
    num_tokens, num_experts = 100, 4

    # Logits heavily favoring expert 0
    router_logits = torch.zeros(num_tokens, num_experts)
    router_logits[:, 0] = 100.0  # Very high logit for expert 0

    # All tokens go to expert 0
    indices = torch.zeros(num_tokens, 1, dtype=torch.long)

    loss = load_balancing_loss(router_logits, indices, num_experts)

    # f_0=1, P_0~1, loss ~ num_experts * 1 * 1 = num_experts
    assert loss.item() > 2.0, f"Skewed loss should be high, got {loss.item()}"


def test_loss_is_non_negative():
    logits = torch.randn(50, 8)
    indices = torch.randint(0, 8, (50, 2))
    loss = load_balancing_loss(logits, indices, num_experts=8)
    assert loss.item() >= 0


def test_loss_increases_with_imbalance():
    """More imbalanced routing should produce higher loss."""
    torch.manual_seed(42)
    num_tokens, num_experts = 200, 8

    # Balanced routing
    balanced_logits = torch.zeros(num_tokens, num_experts)
    balanced_indices = torch.arange(num_tokens).remainder(num_experts).unsqueeze(1)

    # Imbalanced: 80% to expert 0, 20% spread
    imbalanced_logits = torch.zeros(num_tokens, num_experts)
    imbalanced_logits[:, 0] = 5.0
    imbalanced_indices = torch.zeros(num_tokens, 1, dtype=torch.long)
    tail = num_tokens - int(num_tokens * 0.8)
    imbalanced_indices[int(num_tokens * 0.8):, 0] = torch.arange(tail).remainder(num_experts).long()

    loss_balanced = load_balancing_loss(balanced_logits, balanced_indices, num_experts)
    loss_imbalanced = load_balancing_loss(imbalanced_logits, imbalanced_indices, num_experts)

    assert loss_imbalanced > loss_balanced, (
        f"Imbalanced loss ({loss_imbalanced.item():.4f}) should exceed "
        f"balanced loss ({loss_balanced.item():.4f})"
    )
