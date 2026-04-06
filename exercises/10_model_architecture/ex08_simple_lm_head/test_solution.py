"""Tests for Exercise 08: Minimal Language Model"""

import importlib.util
import os
import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SimpleLM = _mod.SimpleLM


def _make_tiny_lm():
    return SimpleLM(
        vocab_size=100,
        dim=64,
        n_layers=2,
        n_heads=4,
        ffn_hidden_dim=128,
        max_seq_len=64,
    )


def test_forward_shape():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (2, 10))
    logits = model(ids)
    assert logits.shape == (2, 10, 100), f"Expected (2, 10, 100), got {logits.shape}"


def test_generate_shape():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (1, 5))
    out = model.generate(ids, max_new_tokens=10)
    assert out.shape == (1, 15), f"Expected (1, 15), got {out.shape}"


def test_generate_preserves_prompt():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (1, 5))
    prompt = ids.clone()
    out = model.generate(ids, max_new_tokens=3)
    assert torch.equal(out[:, :5], prompt), "Generated output should start with the prompt"


def test_generate_tokens_in_vocab():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (1, 3))
    out = model.generate(ids, max_new_tokens=10)
    assert (out >= 0).all() and (out < 100).all(), "Generated tokens should be in [0, vocab_size)"


def test_generate_deterministic():
    """Greedy decoding should be deterministic."""
    torch.manual_seed(42)
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (1, 5))
    out1 = model.generate(ids.clone(), max_new_tokens=5)
    out2 = model.generate(ids.clone(), max_new_tokens=5)
    assert torch.equal(out1, out2), "Greedy generation should be deterministic"


def test_model_has_layers():
    model = _make_tiny_lm()
    assert hasattr(model, 'layers') or hasattr(model, 'blocks'), "Model should have transformer layers"
    # Check layer count
    layers = getattr(model, 'layers', getattr(model, 'blocks', None))
    assert len(layers) == 2, f"Expected 2 layers, got {len(layers)}"


def test_gradient_flows():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (2, 8))
    logits = model(ids)
    loss = logits.sum()
    loss.backward()
    # Check that embedding got gradients
    assert model.tok_emb.weight.grad is not None


def test_single_token():
    model = _make_tiny_lm()
    ids = torch.randint(0, 100, (1, 1))
    logits = model(ids)
    assert logits.shape == (1, 1, 100)
