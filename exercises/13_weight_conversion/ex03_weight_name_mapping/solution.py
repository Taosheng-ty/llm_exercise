"""
Solution for Exercise 03: Map Weight Names Between Frameworks
"""

import re

import numpy as np


def megatron_to_hf_name(megatron_name: str) -> list[str]:
    """Convert a single Megatron-style weight name to HuggingFace-style name(s)."""
    # Non-layer weights
    if megatron_name == "module.module.embedding.word_embeddings.weight":
        return ["model.embed_tokens.weight"]
    if megatron_name == "module.module.output_layer.weight":
        return ["lm_head.weight"]
    if megatron_name == "module.module.decoder.final_layernorm.weight":
        return ["model.norm.weight"]

    # Layer weights
    pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(pattern, megatron_name)
    if not match:
        raise ValueError(f"Unknown parameter name: {megatron_name}")

    layer_idx, rest = match.groups()

    if rest == "self_attention.linear_qkv.weight":
        return [
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        ]
    elif rest == "self_attention.linear_proj.weight":
        return [f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
    elif rest == "mlp.linear_fc1.weight":
        return [
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
        ]
    elif rest == "mlp.linear_fc2.weight":
        return [f"model.layers.{layer_idx}.mlp.down_proj.weight"]
    elif rest == "self_attention.linear_qkv.layer_norm_weight":
        return [f"model.layers.{layer_idx}.input_layernorm.weight"]
    elif rest == "mlp.linear_fc1.layer_norm_weight":
        return [f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
    else:
        raise ValueError(f"Unknown parameter name: {megatron_name}")


def build_full_mapping(num_layers: int) -> dict[str, list[str]]:
    """Build the complete name mapping for a model with num_layers transformer layers."""
    mapping = {}

    # Global weights
    for name in [
        "module.module.embedding.word_embeddings.weight",
        "module.module.output_layer.weight",
        "module.module.decoder.final_layernorm.weight",
    ]:
        mapping[name] = megatron_to_hf_name(name)

    # Per-layer weights
    layer_suffixes = [
        "self_attention.linear_qkv.weight",
        "self_attention.linear_proj.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        "self_attention.linear_qkv.layer_norm_weight",
        "mlp.linear_fc1.layer_norm_weight",
    ]

    for layer_idx in range(num_layers):
        for suffix in layer_suffixes:
            name = f"module.module.decoder.layers.{layer_idx}.{suffix}"
            mapping[name] = megatron_to_hf_name(name)

    return mapping
