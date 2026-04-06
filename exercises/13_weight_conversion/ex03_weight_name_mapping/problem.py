"""
Exercise 03: Map Weight Names Between Frameworks (Medium, numpy)

When converting checkpoints between Megatron-LM and HuggingFace, weight names
must be translated. Each framework uses different naming conventions.

Megatron style:
  module.module.decoder.layers.{L}.self_attention.linear_qkv.weight
  module.module.decoder.layers.{L}.self_attention.linear_proj.weight
  module.module.decoder.layers.{L}.mlp.linear_fc1.weight
  module.module.decoder.layers.{L}.mlp.linear_fc2.weight
  module.module.decoder.layers.{L}.self_attention.linear_qkv.layer_norm_weight
  module.module.decoder.layers.{L}.mlp.linear_fc1.layer_norm_weight
  module.module.embedding.word_embeddings.weight
  module.module.output_layer.weight
  module.module.decoder.final_layernorm.weight

HuggingFace style:
  model.layers.{L}.self_attn.q_proj.weight  (note: QKV splits into 3)
  model.layers.{L}.self_attn.k_proj.weight
  model.layers.{L}.self_attn.v_proj.weight
  model.layers.{L}.self_attn.o_proj.weight
  model.layers.{L}.mlp.gate_proj.weight     (note: fc1 splits into 2)
  model.layers.{L}.mlp.up_proj.weight
  model.layers.{L}.mlp.down_proj.weight
  model.layers.{L}.input_layernorm.weight
  model.layers.{L}.post_attention_layernorm.weight
  model.embed_tokens.weight
  lm_head.weight
  model.norm.weight

Reference: slime/backends/megatron_utils/megatron_to_hf/llama.py

Tasks:
    1. Implement megatron_to_hf_name() that maps a single Megatron name to HF name(s).
    2. Implement build_full_mapping() that generates the complete mapping for N layers.

NOTE: This exercise uses numpy (not torch). The functions work purely with strings.
"""

import numpy as np


def megatron_to_hf_name(megatron_name: str) -> list[str]:
    """Convert a single Megatron-style weight name to HuggingFace-style name(s).

    Some Megatron weights map to multiple HF weights (QKV splits into 3, fc1 splits into 2).
    Returns a list of corresponding HF name(s).

    Args:
        megatron_name: a Megatron-style parameter name

    Returns:
        List of HuggingFace-style parameter names. Typically 1, but 3 for QKV
        and 2 for gate-up (fc1).

    Raises:
        ValueError: if the name is not recognized
    """
    # TODO: Implement this function
    # Hint: use string matching or regex to parse layer index and parameter type
    raise NotImplementedError


def build_full_mapping(num_layers: int) -> dict[str, list[str]]:
    """Build the complete name mapping for a model with num_layers transformer layers.

    Args:
        num_layers: number of transformer layers

    Returns:
        dict mapping each Megatron name to a list of HF names.
        Includes embedding, output layer, final layernorm, and all per-layer weights.
    """
    # TODO: Implement this function
    raise NotImplementedError
