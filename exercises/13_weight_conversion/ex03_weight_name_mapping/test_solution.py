import importlib.util
import os

import pytest

_dir = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("solution", os.path.join(_dir, "solution.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

megatron_to_hf_name = _mod.megatron_to_hf_name
build_full_mapping = _mod.build_full_mapping


class TestMegatronToHfName:
    def test_embedding(self):
        result = megatron_to_hf_name("module.module.embedding.word_embeddings.weight")
        assert result == ["model.embed_tokens.weight"]

    def test_output_layer(self):
        result = megatron_to_hf_name("module.module.output_layer.weight")
        assert result == ["lm_head.weight"]

    def test_final_layernorm(self):
        result = megatron_to_hf_name("module.module.decoder.final_layernorm.weight")
        assert result == ["model.norm.weight"]

    def test_qkv_splits_into_three(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.5.self_attention.linear_qkv.weight"
        )
        assert len(result) == 3
        assert result[0] == "model.layers.5.self_attn.q_proj.weight"
        assert result[1] == "model.layers.5.self_attn.k_proj.weight"
        assert result[2] == "model.layers.5.self_attn.v_proj.weight"

    def test_o_proj(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.0.self_attention.linear_proj.weight"
        )
        assert result == ["model.layers.0.self_attn.o_proj.weight"]

    def test_fc1_splits_into_two(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.3.mlp.linear_fc1.weight"
        )
        assert len(result) == 2
        assert result[0] == "model.layers.3.mlp.gate_proj.weight"
        assert result[1] == "model.layers.3.mlp.up_proj.weight"

    def test_fc2(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.10.mlp.linear_fc2.weight"
        )
        assert result == ["model.layers.10.mlp.down_proj.weight"]

    def test_input_layernorm(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight"
        )
        assert result == ["model.layers.2.input_layernorm.weight"]

    def test_post_attn_layernorm(self):
        result = megatron_to_hf_name(
            "module.module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight"
        )
        assert result == ["model.layers.7.post_attention_layernorm.weight"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            megatron_to_hf_name("some.random.name")

    def test_unknown_layer_suffix_raises(self):
        with pytest.raises(ValueError):
            megatron_to_hf_name("module.module.decoder.layers.0.unknown.weight")


class TestBuildFullMapping:
    def test_single_layer(self):
        mapping = build_full_mapping(1)
        # 3 global + 6 per-layer = 9
        assert len(mapping) == 9
        # Check all expected HF names are present
        all_hf = []
        for hf_names in mapping.values():
            all_hf.extend(hf_names)
        assert "model.embed_tokens.weight" in all_hf
        assert "lm_head.weight" in all_hf
        assert "model.norm.weight" in all_hf
        assert "model.layers.0.self_attn.q_proj.weight" in all_hf

    def test_multi_layer_count(self):
        mapping = build_full_mapping(4)
        # 3 global + 4 * 6 per-layer = 27
        assert len(mapping) == 27

    def test_all_layers_present(self):
        num_layers = 3
        mapping = build_full_mapping(num_layers)
        for i in range(num_layers):
            key = f"module.module.decoder.layers.{i}.self_attention.linear_qkv.weight"
            assert key in mapping
            assert len(mapping[key]) == 3

    def test_hf_names_unique(self):
        """All generated HF names should be unique."""
        mapping = build_full_mapping(8)
        all_hf = []
        for hf_names in mapping.values():
            all_hf.extend(hf_names)
        assert len(all_hf) == len(set(all_hf))
