"""
Tests for Exercise 02: Gradient Accumulation
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

train_with_gradient_accumulation = _mod.train_with_gradient_accumulation


def _make_setup(seed=42, in_features=4, out_features=2):
    torch.manual_seed(seed)
    model = nn.Linear(in_features, out_features)
    return model, in_features, out_features


def test_returns_losses():
    model, inf, outf = _make_setup()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = [(torch.randn(3, inf), torch.randn(3, outf)) for _ in range(6)]
    losses = train_with_gradient_accumulation(model, nn.MSELoss(), optimizer, batches, 3)
    assert len(losses) == 6
    assert all(isinstance(l, float) for l in losses)


def test_loss_is_scaled():
    """Each returned loss should be scaled by 1/accumulation_steps."""
    model, inf, outf = _make_setup()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 so no param update
    batches = [(torch.randn(3, inf), torch.randn(3, outf)) for _ in range(4)]
    accum = 4

    # Compute unscaled losses for comparison
    unscaled_losses = []
    for inputs, targets in batches:
        pred = model(inputs)
        loss = nn.MSELoss()(pred, targets)
        unscaled_losses.append(loss.item())

    losses = train_with_gradient_accumulation(model, nn.MSELoss(), optimizer, batches, accum)
    for scaled, unscaled in zip(losses, unscaled_losses):
        assert abs(scaled - unscaled / accum) < 1e-5


def test_gradient_matches_large_batch():
    """
    Accumulated gradients over micro-batches should match the gradient from a
    single large batch (within floating-point tolerance).
    """
    in_features, out_features = 4, 2
    lr = 0.01

    # Create fixed data
    torch.manual_seed(42)
    all_inputs = torch.randn(8, in_features)
    all_targets = torch.randn(8, out_features)

    # -- Single large batch --
    torch.manual_seed(0)
    model_big = nn.Linear(in_features, out_features)
    opt_big = torch.optim.SGD(model_big.parameters(), lr=lr)
    opt_big.zero_grad()
    pred = model_big(all_inputs)
    loss = nn.MSELoss()(pred, all_targets)
    loss.backward()
    opt_big.step()
    params_big = {n: p.clone() for n, p in model_big.named_parameters()}

    # -- Gradient accumulation (4 micro-batches of size 2, accum_steps=4) --
    torch.manual_seed(0)
    model_acc = nn.Linear(in_features, out_features)
    opt_acc = torch.optim.SGD(model_acc.parameters(), lr=lr)
    micro_batches = [
        (all_inputs[i * 2 : (i + 1) * 2], all_targets[i * 2 : (i + 1) * 2])
        for i in range(4)
    ]
    train_with_gradient_accumulation(model_acc, nn.MSELoss(), opt_acc, micro_batches, 4)
    params_acc = {n: p.clone() for n, p in model_acc.named_parameters()}

    for name in params_big:
        assert torch.allclose(params_big[name], params_acc[name], atol=1e-5), (
            f"Parameter '{name}' differs after gradient accumulation vs large batch"
        )


def test_partial_accumulation():
    """When num_batches is not divisible by accumulation_steps, remaining
    gradients should still be applied."""
    model, inf, outf = _make_setup()
    init_params = {n: p.clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [(torch.randn(3, inf), torch.randn(3, outf)) for _ in range(5)]
    train_with_gradient_accumulation(model, nn.MSELoss(), optimizer, batches, 3)

    # Params should have changed
    changed = False
    for n, p in model.named_parameters():
        if not torch.allclose(init_params[n], p):
            changed = True
            break
    assert changed, "Model parameters should have been updated"


def test_zero_grad_between_steps():
    """After each optimizer step, gradients should be reset."""
    model, inf, outf = _make_setup()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = [(torch.randn(2, inf), torch.randn(2, outf)) for _ in range(6)]
    train_with_gradient_accumulation(model, nn.MSELoss(), optimizer, batches, 3)

    # After the function, gradients should be zeroed (last step done)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad))
