"""
Tests for Exercise 01: Gradient Checkpointing
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

checkpoint_sequential = _mod.checkpoint_sequential
ManualCheckpointFunction = _mod.ManualCheckpointFunction
manual_checkpoint_sequential = _mod.manual_checkpoint_sequential


def _make_layers(num_layers=4, in_features=32):
    """Create a list of simple linear layers with activations."""
    layers = []
    for _ in range(num_layers):
        layers.append(
            nn.Sequential(nn.Linear(in_features, in_features), nn.ReLU())
        )
    return layers


def _plain_sequential(functions, input_tensor):
    """Reference: plain sequential forward without checkpointing."""
    output = input_tensor
    for fn in functions:
        output = fn(output)
    return output


class TestCheckpointSequential:
    def test_output_matches_plain(self):
        """Checkpointed output should match plain sequential output."""
        torch.manual_seed(42)
        layers = _make_layers(4, 32)
        x = torch.randn(8, 32, requires_grad=True)

        # Plain forward
        out_plain = _plain_sequential(layers, x)

        # Checkpointed forward
        x2 = x.detach().clone().requires_grad_(True)
        out_ckpt = checkpoint_sequential(layers, x2)

        torch.testing.assert_close(out_ckpt, out_plain, atol=1e-5, rtol=1e-5)

    def test_gradients_match_plain(self):
        """Gradients through checkpointed forward should match plain."""
        torch.manual_seed(42)
        layers = _make_layers(3, 16)

        x1 = torch.randn(4, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out1 = _plain_sequential(layers, x1)
        loss1 = out1.sum()
        loss1.backward()

        out2 = checkpoint_sequential(layers, x2)
        loss2 = out2.sum()
        loss2.backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)

    def test_single_layer(self):
        """Should work with a single layer."""
        torch.manual_seed(0)
        layers = _make_layers(1, 8)
        x = torch.randn(2, 8, requires_grad=True)
        out = checkpoint_sequential(layers, x)
        assert out.shape == (2, 8)
        out.sum().backward()
        assert x.grad is not None


class TestManualCheckpointFunction:
    def test_output_matches_plain(self):
        """Manual checkpoint output should match plain sequential."""
        torch.manual_seed(42)
        layers = _make_layers(4, 32)
        x = torch.randn(8, 32, requires_grad=True)

        out_plain = _plain_sequential(layers, x)

        x2 = x.detach().clone().requires_grad_(True)
        out_manual = manual_checkpoint_sequential(layers, x2)

        torch.testing.assert_close(out_manual, out_plain, atol=1e-5, rtol=1e-5)

    def test_gradients_flow(self):
        """Backward should produce gradients for the input tensor."""
        torch.manual_seed(42)
        layers = _make_layers(3, 16)
        x = torch.randn(4, 16, requires_grad=True)

        out = manual_checkpoint_sequential(layers, x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradients_match_plain(self):
        """Manual checkpoint gradients should match plain sequential gradients."""
        torch.manual_seed(42)
        layers = _make_layers(3, 16)

        x1 = torch.randn(4, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        out1 = _plain_sequential(layers, x1)
        loss1 = out1.sum()
        loss1.backward()

        out2 = manual_checkpoint_sequential(layers, x2)
        loss2 = out2.sum()
        loss2.backward()

        torch.testing.assert_close(x1.grad, x2.grad, atol=1e-4, rtol=1e-4)

    def test_layer_params_get_gradients(self):
        """Layer parameters should also receive gradients through manual checkpoint."""
        torch.manual_seed(42)
        layers = _make_layers(2, 8)
        x = torch.randn(2, 8, requires_grad=True)

        out = manual_checkpoint_sequential(layers, x)
        loss = out.sum()
        loss.backward()

        for layer in layers:
            for param in layer.parameters():
                assert param.grad is not None, "Layer parameters should receive gradients"


class TestBothMethodsAgree:
    def test_torch_and_manual_produce_same_output(self):
        """Both checkpoint methods should produce the same output."""
        torch.manual_seed(42)
        layers = _make_layers(4, 32)
        x = torch.randn(8, 32, requires_grad=True)

        x1 = x.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)

        out1 = checkpoint_sequential(layers, x1)
        out2 = manual_checkpoint_sequential(layers, x2)

        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)
