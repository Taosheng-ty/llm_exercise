"""
Tests for Exercise 04: CPU Offloading
"""

import importlib.util
import os

import torch
import torch.nn as nn

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OffloadWrapper = _mod.OffloadWrapper
offload_forward_pass = _mod.offload_forward_pass


def _get_param_device(module):
    """Get the device of the first parameter."""
    for p in module.parameters():
        return str(p.device)
    return None


class TestOffloadWrapper:
    def test_initial_offload(self):
        """After wrapping, parameters should be on CPU."""
        model = nn.Linear(16, 8)
        wrapper = OffloadWrapper(model, device="cpu")
        assert _get_param_device(wrapper.module) == "cpu"

    def test_onload_and_offload(self):
        """onload should move to compute device, offload back to CPU."""
        model = nn.Linear(16, 8)
        wrapper = OffloadWrapper(model, device="cpu")

        # onload to CPU (since CUDA may not be available)
        wrapper.onload("cpu")
        assert _get_param_device(wrapper.module) == "cpu"

        # offload
        wrapper.offload()
        assert _get_param_device(wrapper.module) == "cpu"

    def test_forward_produces_correct_output(self):
        """Forward should produce same output as unwrapped module."""
        torch.manual_seed(42)
        model = nn.Linear(16, 8)
        x = torch.randn(4, 16)

        # Reference output
        with torch.no_grad():
            ref_output = model(x)

        # Wrapped output
        wrapper = OffloadWrapper(model, device="cpu")
        with torch.no_grad():
            wrapped_output = wrapper(x)

        torch.testing.assert_close(wrapped_output, ref_output)

    def test_params_offloaded_after_forward(self):
        """After forward, parameters should be back on offload device."""
        model = nn.Linear(16, 8)
        wrapper = OffloadWrapper(model, device="cpu")
        x = torch.randn(4, 16)

        with torch.no_grad():
            wrapper(x)

        assert _get_param_device(wrapper.module) == "cpu"

    def test_chaining(self):
        """offload() and onload() should return self for chaining."""
        model = nn.Linear(8, 4)
        wrapper = OffloadWrapper(model)
        result = wrapper.offload()
        assert result is wrapper
        result = wrapper.onload("cpu")
        assert result is wrapper


class TestOffloadForwardPass:
    def test_correct_output(self):
        """Output should match sequential forward without offloading."""
        torch.manual_seed(42)
        layers = nn.ModuleList([
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 8),
        ])
        x = torch.randn(4, 16)

        # Reference: plain sequential
        with torch.no_grad():
            ref = x.clone()
            for layer in layers:
                ref = layer(ref)

        # Offloaded forward
        with torch.no_grad():
            out = offload_forward_pass(list(layers), x, compute_device="cpu")

        torch.testing.assert_close(out, ref)

    def test_layers_offloaded_after(self):
        """All layers should be on CPU after offload_forward_pass."""
        layers = [nn.Linear(8, 8) for _ in range(3)]
        x = torch.randn(2, 8)

        with torch.no_grad():
            offload_forward_pass(layers, x, compute_device="cpu")

        for layer in layers:
            assert _get_param_device(layer) == "cpu"

    def test_single_layer(self):
        """Should work with a single layer."""
        torch.manual_seed(0)
        layer = nn.Linear(4, 4)
        x = torch.randn(2, 4)

        with torch.no_grad():
            ref = layer(x)

        with torch.no_grad():
            out = offload_forward_pass([layer], x, compute_device="cpu")

        torch.testing.assert_close(out, ref)

    def test_empty_layers(self):
        """Should handle empty layer list by returning input."""
        x = torch.randn(2, 4)
        out = offload_forward_pass([], x, compute_device="cpu")
        torch.testing.assert_close(out, x)

    def test_multiple_forward_passes(self):
        """Should produce consistent results across multiple calls."""
        torch.manual_seed(42)
        layers = [nn.Linear(8, 8), nn.Linear(8, 4)]
        x = torch.randn(2, 8)

        with torch.no_grad():
            out1 = offload_forward_pass(layers, x, compute_device="cpu")
            out2 = offload_forward_pass(layers, x, compute_device="cpu")

        torch.testing.assert_close(out1, out2)
