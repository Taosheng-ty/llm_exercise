import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SharedMoELayer = _mod.SharedMoELayer


def test_output_shape():
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2)
    x = torch.randn(8, 16)
    out = layer(x)
    assert out.shape == (8, 16)


def test_shared_expert_always_contributes():
    """Even with zero-initialized routed experts, output should be non-zero
    due to the shared expert."""
    torch.manual_seed(42)
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2)

    # Zero out all routed experts
    for expert in layer.routed_experts:
        for param in expert.parameters():
            param.data.zero_()

    x = torch.randn(4, 16)
    out = layer(x)

    # Output should equal shared expert output
    shared_out = layer.shared_expert(x)
    assert torch.allclose(out, shared_out, atol=1e-5)


def test_routed_experts_contribute():
    """Output should differ from shared-only when routed experts are active."""
    torch.manual_seed(0)
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2)

    x = torch.randn(4, 16)
    full_out = layer(x)
    shared_only = layer.shared_expert(x)

    # If routed experts have non-zero weights, outputs should differ
    assert not torch.allclose(full_out, shared_only, atol=1e-3), (
        "Routed experts should contribute to the output"
    )


def test_has_router():
    """Layer should have a router attribute."""
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2)
    assert hasattr(layer, "router"), "Layer must have a router"


def test_correct_number_of_experts():
    layer = SharedMoELayer(dim=16, num_routed_experts=6, top_k=2)
    assert len(layer.routed_experts) == 6


def test_gradient_flows():
    """Ensure gradients flow through both shared and routed experts."""
    torch.manual_seed(42)
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2)

    x = torch.randn(8, 16, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Check gradient flows to input
    assert x.grad is not None
    assert not torch.all(x.grad == 0)

    # Check gradient flows to shared expert
    for param in layer.shared_expert.parameters():
        assert param.grad is not None

    # Check gradient flows to router
    assert layer.router.weight.grad is not None


def test_custom_intermediate_dim():
    layer = SharedMoELayer(dim=16, num_routed_experts=4, top_k=2, intermediate_dim=32)
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 16)
