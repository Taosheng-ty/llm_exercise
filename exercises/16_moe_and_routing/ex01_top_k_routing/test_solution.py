import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

top_k_routing = _mod.top_k_routing


def test_output_shapes():
    num_tokens, dim, num_experts, top_k = 8, 16, 4, 2
    hidden = torch.randn(num_tokens, dim)
    router_w = torch.randn(dim, num_experts)

    weights, indices = top_k_routing(hidden, router_w, top_k)

    assert weights.shape == (num_tokens, top_k)
    assert indices.shape == (num_tokens, top_k)


def test_weights_sum_to_one():
    num_tokens, dim, num_experts, top_k = 16, 32, 8, 2
    hidden = torch.randn(num_tokens, dim)
    router_w = torch.randn(dim, num_experts)

    weights, _ = top_k_routing(hidden, router_w, top_k)

    # Each row should sum to 1 (softmax over top-k)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(num_tokens), atol=1e-5)


def test_weights_non_negative():
    hidden = torch.randn(10, 16)
    router_w = torch.randn(16, 6)
    weights, _ = top_k_routing(hidden, router_w, 3)
    assert (weights >= 0).all()


def test_indices_in_valid_range():
    num_experts = 8
    hidden = torch.randn(10, 16)
    router_w = torch.randn(16, num_experts)
    _, indices = top_k_routing(hidden, router_w, 2)

    assert (indices >= 0).all()
    assert (indices < num_experts).all()


def test_top_k_selects_highest_logits():
    """Verify that top-k selects the experts with highest router logits."""
    num_tokens, dim, num_experts = 4, 8, 6
    top_k = 2
    hidden = torch.randn(num_tokens, dim)
    router_w = torch.randn(dim, num_experts)

    weights, indices = top_k_routing(hidden, router_w, top_k)

    # Manually compute logits and verify
    logits = hidden @ router_w
    expected_indices = torch.topk(logits, top_k, dim=-1).indices

    assert torch.equal(indices, expected_indices)


def test_load_balance_with_uniform_router():
    """With a near-identity router, experts should be selected roughly uniformly."""
    torch.manual_seed(42)
    num_tokens, dim, num_experts, top_k = 1000, 8, 8, 2

    # Random hidden states with small router weights -> near-uniform routing
    hidden = torch.randn(num_tokens, dim)
    router_w = torch.randn(dim, num_experts) * 0.01

    _, indices = top_k_routing(hidden, router_w, top_k)

    # Count per-expert assignments
    counts = torch.zeros(num_experts)
    for e in range(num_experts):
        counts[e] = (indices == e).sum().float()

    expected_per_expert = num_tokens * top_k / num_experts
    # Each expert should get roughly expected_per_expert assignments
    for e in range(num_experts):
        assert counts[e] > expected_per_expert * 0.3, (
            f"Expert {e} got {counts[e]:.0f} tokens, expected ~{expected_per_expert:.0f}"
        )
