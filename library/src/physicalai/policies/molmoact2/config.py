# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for MolmoAct2 model.

This module provides a dataclass configuration for the MolmoAct2
vision-language-action model in physicalai format.

Example (CLI):
    physicalai fit --config configs/physicalai/molmoact2.yaml

Example (API):
    >>> from physicalai.policies.molmoact2 import MolmoAct2Config
    >>> config = MolmoAct2Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from physicalai.config import Config

from physicalai.data import Feature  # noqa: TC001


@dataclass
class MolmoAct2Config(Config):
    """Flat configuration for the native MolmoAct2 model and policy."""

    # Policy arguments
    input_features: list[Feature] | None = field(default_factory=list)
    output_features: list[Feature] | None = field(default_factory=list)
    norm_tag: str | None = None
    n_action_steps: int = 30
    chunk_size: int = 30
    n_obs_steps: int = 1
    setup_type: str = ""
    control_mode: str = ""
    adapt_to_so101: bool = False

    # Text transformer
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    vocab_size: int = 154_624
    additional_vocab_size: int = 128
    qkv_bias: bool = False
    num_hidden_layers: int = 36
    intermediate_size: int = 9728
    hidden_act: str = "silu"
    max_position_embeddings: int = 16_384
    rope_theta: float = 5_000_000.0
    use_qk_norm: bool = True
    layer_norm_eps: float = 1e-6
    norm_after: bool = False
    use_cache: bool = True
    text_attn_implementation: str = "sdpa"

    # Vision transformer
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_hidden_layers: int = 27
    vision_num_attention_heads: int = 16
    vision_num_key_value_heads: int = 16
    vision_head_dim: int = 72
    vision_hidden_act: str = "gelu_pytorch_tanh"
    vision_layer_norm_eps: float = 1e-6
    image_default_input_size: tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 729
    vision_attention_dropout: float = 0.0
    vision_residual_dropout: float = 0.0
    vision_attn_implementation: str = "sdpa"

    # Vision adapter
    adapter_vit_layers: tuple[int, ...] = (-3, -9)
    adapter_pooling_attention_mask: bool = True
    adapter_hidden_size: int = 1152
    adapter_num_attention_heads: int = 16
    adapter_num_key_value_heads: int = 16
    adapter_head_dim: int = 72
    adapter_attention_dropout: float = 0.0
    adapter_residual_dropout: float = 0.0
    adapter_hidden_act: str = "silu"
    adapter_intermediate_size: int = 9728
    adapter_text_hidden_size: int = 2560
    image_feature_dropout: float = 0.0
    adapter_attn_implementation: str = "sdpa"

    # Action expert
    action_expert_max_action_horizon: int = 30
    action_expert_max_action_dim: int = 32
    action_expert_hidden_size: int = 768
    action_expert_num_layers: int = 36
    action_expert_num_heads: int = 8
    action_expert_mlp_ratio: float = 4.0
    action_expert_ffn_multiple_of: int = 256
    action_expert_timestep_embed_dim: int = 256
    action_expert_context_layer_norm: bool = True
    action_expert_qk_norm: bool = True
    action_expert_qk_norm_eps: float = 1e-6
    action_expert_rope: bool = True
    action_expert_causal_attn: bool = False
    add_action_expert: bool = True

    # Action structure
    max_action_dim: int = 32
    action_mode: Literal["continuous", "discrete", "both"] = "continuous"
    state_format: Literal["discrete"] = "discrete"

    # Flow matching
    flow_matching_num_steps: int = 10
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    mask_action_dim_padding: bool = True
    use_random_input_noise: bool = False

    # Token and prompt layout
    num_state_tokens: int = 256
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    image_start_token_id: int | None = 154624
    image_end_token_id: int | None = 154625
    image_patch_id: int | None = 154626
    image_col_id: int | None = 154627
    low_res_image_start_token_id: int | None = 154628
    image_placeholder_token_id: int = 154629
    image_low_res_id: int | None = 154630
    frame_start_token_id: int | None = 154631
    frame_end_token_id: int | None = 154632

    # Checkpoint
    checkpoint_path: str | None = None

    # LoRA parameters
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: Literal["all", "lora_only", "none"] = "none"

    # Tokenizer
    tokenizer_name_or_path: str = "allenai/MolmoAct2"
    tokenizer_revision: str | None = None
    tokenizer_max_length: int = 256
    tokenizer_padding: Literal["max_length", "longest"] = "max_length"
    tokenizer_config: dict[str, Any] | None = None

    # Image processor
    image_processor_crop_mode: str = "resize"
    image_processor_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_processor_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_processor_patch_size: int = 14
    image_processor_pooling_size: list[int] = field(default_factory=lambda: [2, 2])
    image_processor_size: dict[str, int] = field(default_factory=lambda: {"height": 378, "width": 378})
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool | None = False
    use_single_crop_start_token: bool = True

    # Normalization and feature processing
    normalization_mode: str = "QUANTILES"
    normalize_gripper: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self._validate_rollout_settings()

    def _validate_rollout_settings(self) -> None:
        if self.chunk_size < 1:
            msg = f"chunk_size must be >= 1, got {self.chunk_size}"
            raise ValueError(msg)
        if self.n_action_steps < 1:
            msg = f"n_action_steps must be >= 1, got {self.n_action_steps}"
            raise ValueError(msg)
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)
        if self.n_obs_steps < 1:
            msg = f"n_obs_steps must be >= 1, got {self.n_obs_steps}"
            raise ValueError(msg)
        if self.max_action_dim < 1:
            msg = f"max_action_dim must be >= 1, got {self.max_action_dim}"
            raise ValueError(msg)
        if self.lora_rank < 1:
            msg = f"MolmoAct2 lora_rank must be >= 1, got {self.lora_rank}."
            raise ValueError(msg)
        if not 0.0 <= self.lora_dropout < 1.0:
            msg = f"MolmoAct2 lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}."
            raise ValueError(msg)
        if self.lora_bias not in {"none", "all", "lora_only"}:
            msg = f"MolmoAct2 lora_bias must be one of 'none', 'all', 'lora_only', got {self.lora_bias!r}."
            raise ValueError(msg)
