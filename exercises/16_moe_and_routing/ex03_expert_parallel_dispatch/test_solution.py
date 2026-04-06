import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

expert_dispatch = _mod.expert_dispatch


def _make_expert_ffns(num_experts, dim):
    """Create simple linear expert FFNs."""
    experts = []
    for _ in range(num_experts):
        ffn = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(ffn.weight, std=0.1)
        experts.append(ffn)
    return experts


def test_output_shape():
    num_tokens, dim, num_experts, top_k = 8, 16, 4, 2
    hidden = torch.randn(num_tokens, dim)
    weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)
    indices = torch.randint(0, num_experts, (num_tokens, top_k))
    experts = _make_expert_ffns(num_experts, dim)

    output = expert_dispatch(hidden, weights, indices, experts, num_experts)
    assert output.shape == (num_tokens, dim)


def test_single_expert_top1():
    """With top_k=1 and one expert, output should be weight * expert(input)."""
    dim = 8
    hidden = torch.randn(4, dim)
    expert = nn.Linear(dim, dim, bias=False)

    weights = torch.ones(4, 1)
    indices = torch.zeros(4, 1, dtype=torch.long)

    output = expert_dispatch(hidden, weights, indices, [expert], num_experts=1)
    expected = expert(hidden)

    assert torch.allclose(output, expected, atol=1e-5)


def test_matches_sequential_processing():
    """Verify dispatch matches naive sequential expert processing."""
    torch.manual_seed(42)
    num_tokens, dim, num_experts, top_k = 6, 8, 3, 2

    hidden = torch.randn(num_tokens, dim)
    experts = _make_expert_ffns(num_experts, dim)

    # Deterministic routing
    indices = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 0], [1, 0], [2, 1]])
    weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    # Dispatch result
    output = expert_dispatch(hidden, weights, indices, experts, num_experts)

    # Sequential reference
    expected = torch.zeros_like(hidden)
    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = indices[token_idx, k].item()
            w = weights[token_idx, k]
            expert_out = experts[expert_id](hidden[token_idx:token_idx + 1])
            expected[token_idx] += w * expert_out.squeeze(0)

    assert torch.allclose(output, expected, atol=1e-5), (
        f"Max diff: {(output - expected).abs().max().item()}"
    )


def test_handles_empty_expert():
    """An expert that receives no tokens should not cause errors."""
    dim = 8
    hidden = torch.randn(4, dim)
    experts = _make_expert_ffns(4, dim)

    # All tokens go to expert 0 and 1, experts 2 and 3 get nothing
    indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    weights = torch.softmax(torch.randn(4, 2), dim=-1)

    output = expert_dispatch(hidden, weights, indices, experts, num_experts=4)
    assert output.shape == (4, dim)
    assert not torch.isnan(output).any()


def test_weights_affect_output():
    """Changing routing weights should change the output."""
    torch.manual_seed(0)
    dim = 8
    hidden = torch.randn(4, dim)
    experts = _make_expert_ffns(2, dim)
    indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])

    w1 = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    w2 = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])

    out1 = expert_dispatch(hidden, w1, indices, experts, num_experts=2)
    out2 = expert_dispatch(hidden, w2, indices, experts, num_experts=2)

    assert not torch.allclose(out1, out2, atol=1e-3), "Different weights should produce different outputs"
