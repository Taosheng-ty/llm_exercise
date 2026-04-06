import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

split_gate_up = _mod.split_gate_up
fuse_gate_up = _mod.fuse_gate_up
split_moe_gate_up = _mod.split_moe_gate_up


class TestSplitGateUp:
    def test_basic_split(self):
        intermediate, hidden = 256, 128
        fused = torch.randn(2 * intermediate, hidden)
        gate, up = split_gate_up(fused)
        assert gate.shape == (intermediate, hidden)
        assert up.shape == (intermediate, hidden)

    def test_split_values(self):
        fused = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        gate, up = split_gate_up(fused)
        assert torch.equal(gate, fused[:3])
        assert torch.equal(up, fused[3:])

    def test_roundtrip(self):
        fused = torch.randn(512, 256)
        gate, up = split_gate_up(fused)
        reconstructed = fuse_gate_up(gate, up)
        assert torch.equal(fused, reconstructed)

    def test_fuse_shapes(self):
        gate = torch.randn(128, 64)
        up = torch.randn(128, 64)
        fused = fuse_gate_up(gate, up)
        assert fused.shape == (256, 64)

    def test_moe_split(self):
        intermediate, hidden = 64, 32
        experts = {
            0: torch.randn(2 * intermediate, hidden),
            1: torch.randn(2 * intermediate, hidden),
            5: torch.randn(2 * intermediate, hidden),
        }
        result = split_moe_gate_up(experts)

        assert set(result.keys()) == {0, 1, 5}
        for eid in experts:
            gate, up = result[eid]
            assert gate.shape == (intermediate, hidden)
            assert up.shape == (intermediate, hidden)
            # Verify values match
            reconstructed = fuse_gate_up(gate, up)
            assert torch.equal(reconstructed, experts[eid])

    def test_moe_empty(self):
        result = split_moe_gate_up({})
        assert result == {}
