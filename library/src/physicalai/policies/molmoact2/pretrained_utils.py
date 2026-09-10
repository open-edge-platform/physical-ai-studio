# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for flattening pretrained MolmoAct2 configuration data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TEXT_CONFIG_MAP = {
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_attention_heads",
    "num_key_value_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "vocab_size": "vocab_size",
    "additional_vocab_size": "additional_vocab_size",
    "qkv_bias": "qkv_bias",
    "num_hidden_layers": "num_hidden_layers",
    "intermediate_size": "intermediate_size",
    "hidden_act": "hidden_act",
    "max_position_embeddings": "max_position_embeddings",
    "rope_theta": "rope_theta",
    "use_qk_norm": "use_qk_norm",
    "layer_norm_eps": "layer_norm_eps",
    "norm_after": "norm_after",
    "use_cache": "use_cache",
    "attn_implementation": "text_attn_implementation",
}

VISION_CONFIG_MAP = {
    "hidden_size": "vision_hidden_size",
    "intermediate_size": "vision_intermediate_size",
    "num_hidden_layers": "vision_num_hidden_layers",
    "num_attention_heads": "vision_num_attention_heads",
    "num_key_value_heads": "vision_num_key_value_heads",
    "head_dim": "vision_head_dim",
    "hidden_act": "vision_hidden_act",
    "layer_norm_eps": "vision_layer_norm_eps",
    "image_default_input_size": "image_default_input_size",
    "image_patch_size": "image_patch_size",
    "image_num_pos": "image_num_pos",
    "attention_dropout": "vision_attention_dropout",
    "residual_dropout": "vision_residual_dropout",
    "attn_implementation": "vision_attn_implementation",
}

ADAPTER_CONFIG_MAP = {
    "vit_layers": "adapter_vit_layers",
    "pooling_attention_mask": "adapter_pooling_attention_mask",
    "hidden_size": "adapter_hidden_size",
    "num_attention_heads": "adapter_num_attention_heads",
    "num_key_value_heads": "adapter_num_key_value_heads",
    "head_dim": "adapter_head_dim",
    "attention_dropout": "adapter_attention_dropout",
    "residual_dropout": "adapter_residual_dropout",
    "hidden_act": "adapter_hidden_act",
    "intermediate_size": "adapter_intermediate_size",
    "text_hidden_size": "adapter_text_hidden_size",
    "image_feature_dropout": "image_feature_dropout",
    "attn_implementation": "adapter_attn_implementation",
}

ACTION_EXPERT_CONFIG_MAP = {
    "max_action_horizon": "action_expert_max_action_horizon",
    "max_action_dim": "action_expert_max_action_dim",
    "hidden_size": "action_expert_hidden_size",
    "num_layers": "action_expert_num_layers",
    "num_heads": "action_expert_num_heads",
    "mlp_ratio": "action_expert_mlp_ratio",
    "ffn_multiple_of": "action_expert_ffn_multiple_of",
    "timestep_embed_dim": "action_expert_timestep_embed_dim",
    "context_layer_norm": "action_expert_context_layer_norm",
    "qk_norm": "action_expert_qk_norm",
    "qk_norm_eps": "action_expert_qk_norm_eps",
    "rope": "action_expert_rope",
    "causal_attn": "action_expert_causal_attn",
}

TOP_LEVEL_CONFIG_MAP = {
    "action_mode": "action_mode",
    "add_action_expert": "add_action_expert",
    "add_control_tokens": "add_control_tokens",
    "add_setup_tokens": "add_setup_tokens",
    "flow_matching_beta_alpha": "flow_matching_beta_alpha",
    "flow_matching_beta_beta": "flow_matching_beta_beta",
    "flow_matching_cutoff": "flow_matching_cutoff",
    "flow_matching_num_steps": "flow_matching_num_steps",
    "flow_matching_time_offset": "flow_matching_time_offset",
    "flow_matching_time_scale": "flow_matching_time_scale",
    "frame_end_token_id": "frame_end_token_id",
    "frame_start_token_id": "frame_start_token_id",
    "image_col_id": "image_col_id",
    "image_end_token_id": "image_end_token_id",
    "image_low_res_id": "image_low_res_id",
    "image_patch_id": "image_patch_id",
    "image_start_token_id": "image_start_token_id",
    "low_res_image_start_token_id": "low_res_image_start_token_id",
    "mask_action_dim_padding": "mask_action_dim_padding",
    "max_action_dim": "max_action_dim",
    "n_action_steps": "n_action_steps",
    "n_obs_steps": "n_obs_steps",
    "num_flow_timesteps": "num_flow_timesteps",
    "num_state_tokens": "num_state_tokens",
    "state_format": "state_format",
    "use_random_input_noise": "use_random_input_noise",
}


def copy_component(
    hf_config: Mapping[str, Any],
    flat_config: dict[str, Any],
    component_name: str | None,
    field_map: Mapping[str, str],
) -> None:
    """Copy pretrained fields from the root or one nested component.

    Raises:
        TypeError: If the selected component is not a mapping.
    """
    component = hf_config if component_name is None else hf_config.get(component_name)
    if component is None:
        return
    if not isinstance(component, Mapping):
        msg = f"Invalid {component_name or 'top-level config'}: expected a mapping."
        raise TypeError(msg)

    for source_key, target_key in field_map.items():
        if source_key in component:
            flat_config[target_key] = component[source_key]
