import importlib.util
import os

import torch

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

merge_lora = _mod.merge_lora
unmerge_lora = _mod.unmerge_lora
merge_lora_state_dict = _mod.merge_lora_state_dict


class TestMergeLora:
    def test_basic_merge(self):
        out_f, in_f, r = 64, 32, 4
        base = torch.randn(out_f, in_f)
        A = torch.randn(r, in_f)
        B = torch.randn(out_f, r)
        alpha = 8.0

        merged = merge_lora(base, A, B, alpha, r)
        expected = base + (alpha / r) * (B @ A)
        assert torch.allclose(merged, expected)

    def test_shape_preserved(self):
        base = torch.randn(128, 256)
        A = torch.randn(8, 256)
        B = torch.randn(128, 8)
        merged = merge_lora(base, A, B, 16.0, 8)
        assert merged.shape == (128, 256)

    def test_alpha_equals_r_no_scaling(self):
        base = torch.zeros(4, 4)
        A = torch.eye(4)[:2]   # (2, 4)
        B = torch.eye(4)[:, :2]  # (4, 2)
        merged = merge_lora(base, A, B, alpha=2.0, r=2)
        # scaling = 1.0, so merged = B @ A
        expected = B @ A
        assert torch.allclose(merged, expected)

    def test_zero_lora_no_change(self):
        base = torch.randn(16, 16)
        A = torch.zeros(4, 16)
        B = torch.zeros(16, 4)
        merged = merge_lora(base, A, B, 8.0, 4)
        assert torch.allclose(merged, base)


class TestUnmergeLora:
    def test_roundtrip(self):
        out_f, in_f, r = 32, 16, 4
        alpha = 8.0
        base = torch.randn(out_f, in_f)
        A = torch.randn(r, in_f)
        B = torch.randn(out_f, r)

        merged = merge_lora(base, A, B, alpha, r)
        delta = unmerge_lora(merged, base, alpha, r)

        expected_delta = (alpha / r) * (B @ A)
        assert torch.allclose(delta, expected_delta, atol=1e-6)

    def test_zero_delta(self):
        base = torch.randn(8, 8)
        delta = unmerge_lora(base, base, 1.0, 1)
        assert torch.allclose(delta, torch.zeros_like(base), atol=1e-7)


class TestMergeLoraStateDict:
    def test_single_layer(self):
        base_sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
            "model.layers.0.self_attn.q_proj.bias": torch.randn(64),
        }
        lora_sd = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 32),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(64, 4),
        }

        merged = merge_lora_state_dict(base_sd, lora_sd, alpha=8.0, r=4)

        assert set(merged.keys()) == set(base_sd.keys())
        # The weight should be modified
        assert not torch.equal(merged["model.layers.0.self_attn.q_proj.weight"],
                               base_sd["model.layers.0.self_attn.q_proj.weight"])
        # The bias should be unchanged (clone)
        assert torch.equal(merged["model.layers.0.self_attn.q_proj.bias"],
                           base_sd["model.layers.0.self_attn.q_proj.bias"])

    def test_no_lora_unchanged(self):
        base_sd = {"w": torch.randn(8, 8)}
        merged = merge_lora_state_dict(base_sd, {}, alpha=1.0, r=1)
        assert torch.equal(merged["w"], base_sd["w"])

    def test_multiple_layers(self):
        base_sd = {}
        lora_sd = {}
        for i in range(3):
            base_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(32, 16)
            lora_sd[f"model.layers.{i}.self_attn.q_proj.lora_A.weight"] = torch.randn(4, 16)
            lora_sd[f"model.layers.{i}.self_attn.q_proj.lora_B.weight"] = torch.randn(32, 4)

        merged = merge_lora_state_dict(base_sd, lora_sd, alpha=8.0, r=4)

        for i in range(3):
            key = f"model.layers.{i}.self_attn.q_proj.weight"
            assert not torch.equal(merged[key], base_sd[key])

    def test_partial_lora(self):
        """Only some weights have LoRA, others should be unchanged."""
        base_sd = {
            "layer.0.weight": torch.randn(16, 8),
            "layer.1.weight": torch.randn(16, 8),
        }
        lora_sd = {
            "layer.0.lora_A.weight": torch.randn(2, 8),
            "layer.0.lora_B.weight": torch.randn(16, 2),
        }

        merged = merge_lora_state_dict(base_sd, lora_sd, alpha=4.0, r=2)

        # layer.0 should be merged
        assert not torch.equal(merged["layer.0.weight"], base_sd["layer.0.weight"])
        # layer.1 should be unchanged
        assert torch.equal(merged["layer.1.weight"], base_sd["layer.1.weight"])
