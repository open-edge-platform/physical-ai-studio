# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
"""RLDX network configuration dataclass."""

from dataclasses import dataclass, field
from typing import Literal

from physicalai.config import Config


@dataclass
class Rldx1Config(Config):
    """Unified configuration for RLDX model with backbone and action model."""

    dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone architecture
    base_model_path: str | None = "RLWRLD/RLDX-1-PT"

    revision: str | None = None
    backbone_embedding_dim: int = 4096  # project_to_dim
    select_layer: int = 18
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = True  # Enable BF16 loading
    freeze_cog_tokens: bool = False  # Freeze cog_emb to prevent VLM backprop

    # Backbone fine-tuning control
    tune_top_llm_layers: int = 4  # Number of top LLM layers to tune
    tune_llm: bool = False
    tune_visual: bool = False
    # TODO @maintainer: upstream defaults this to True, but the fp32 copies of  # noqa: FIX002, TD003
    # trainable backbone params OOM on an A100. DeepSpeed ZeRO-Offload (CPU)
    # avoids the OOM but is very slow in practice. Explore a better way to
    # re-enable True by default.
    backbone_trainable_params_fp32: bool = False

    # Backbone (Qwen3 LLM) LoRA. Mirror of the action-model surface:
    # ``backbone_use_lora`` toggles PEFT injection into the LLM layers;
    # ``backbone_lora_num_layers`` picks the top-N suffix (-1 = all layers,
    # 0 = skip, N > 0 = last N). When LoRA is active the backbone is set
    # to ``requires_grad_(False)`` first and only the injected LoRA params
    # remain trainable — so ``tune_top_llm_layers`` is effectively ignored
    # (the launcher warns about the conflict).
    # ``backbone_lora_target_modules`` covers Qwen3 attention + MLP
    # projections.
    backbone_use_lora: bool = False
    backbone_lora_rank: int = 64
    backbone_lora_alpha: int = 64
    backbone_lora_dropout: float = 0.0
    backbone_lora_num_layers: int = -1
    backbone_lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Image pipeline parameters
    # Aspect-ratio-preserving resize + m-aligned crop (torchvision).
    image_max_area: int = 65536  # 256 * 256 by default
    image_resize_m: int = 32
    image_min_area: int | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = False  # Global flag to enable per-embodiment sin/cos encoding
    use_percentiles: bool = True
    conversation_image_first: bool = False

    # Video input configuration
    # ``use_video`` is an architectural invariant: every supported
    # checkpoint embeds VTC video tokens.
    use_video: bool = True
    video_length: int = 4
    video_stride: int = 2  # Action-step stride between video frames in context window

    # Action head architecture (MSAT)
    max_state_dim: int = 64  # Default from state_shape
    max_action_dim: int = 64  # Default from action_shape
    action_horizon: int = 16
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    general_embodiment_train_ratio: float = 0
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = False
    max_seq_len: int = 1024
    n_cog_tokens: int = 64
    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "attention_head_dim": 64,
            "depth_multi_stream": 4,
            "depth_single_stream": 8,
            "dropout": 0.2,
            "num_attention_heads": 24,
            "output_dim": 1024,
            "positional_embeddings": "rope_sa_only",
            "rope_theta": 10000.0,
            "temb_type": "input_token",
            "gradient_checkpointing": False,
            "action_model_max_seq_len": 512,
            "pre_norm": "layer_norm",
            "qk_norm": "rms_norm",
        },
    )

    # Action head fine-tuning control
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # Action model (MSAT) LoRA. When ``action_model_use_lora=True``,
    # ``RLDXActionModel.set_trainable_parameters`` injects PEFT LoRA
    # adapters into the MSAT linear projections listed in
    # ``action_model_lora_target_modules`` instead of full-tuning the DiT.
    # The default target list covers MSAT's V-L / state-action / physics
    # QKV + output projections + the MMDiT inner FFN linears (see
    # ``rldx/model/modules/action_model/blocks.py``). Targets that don't
    # exist in the current MSAT (e.g. ``p_qkv``/``p_proj`` when
    # ``use_physics=False``) are filtered before the PEFT call.
    action_model_use_lora: bool = False
    action_model_lora_rank: int = 64
    action_model_lora_alpha: int = 64
    action_model_lora_dropout: float = 0.0
    action_model_lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "vl_qkv",
            "vl_proj",
            "sa_qkv",
            "sa_proj",
            "p_qkv",
            "p_proj",
            "linear1",
            "linear2",
        ],
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # State Augmentation parameters
    state_dropout_prob: float = 0.0  # State dropout probability
    state_additive_noise_scale: float = 0.0  # Scale for additive Gaussian noise on state features
    clip_outliers: bool = True  # Studio-only: gates train-time clip + inference clamp

    # Multi-embodiment parameters
    max_num_embodiments: int = 36
    embodiment_id: int | str = 0  # Studio-only: resolved to a projector slot int by Rldx1Config
    embodiment_tag: str = "general_embodiment"  # Studio-only: resolved to a projector slot int by Rldx1Config

    # Memory configuration (phase-2 add-on)
    use_memory: bool = False  # Enable memory-augmented cognition tokens
    memory_length: int = 4  # Number of past timesteps for memory (= context_window)
    memory_n_cog_tokens: int | None = (
        None  # Number of cognition tokens routed through memory (defaults to n_cog_tokens)
    )
    concat_memory: bool = False  # If True, concat MQ_augmented after MQ_original instead of replacing
    memory_dropout_prob: float = 0.0  # Dropout ratio for augmented cognition tokens (concat_memory=True only, mask-out)
    memory_stride: int = 16  # Action-step stride between memory snapshots (should match execution_horizon)
    memory_cfg: dict = field(
        default_factory=lambda: {
            "hidden_size": 4096,
            "intermediate_size": 16384,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5,
            "use_causal_attn": True,
            "use_rope": True,
        },
    )

    # Motion module configuration (phase-2 add-on)
    use_motion: bool = False
    motion_insert_layer: int = 9
    motion_d_hid: int = 512
    motion_window: tuple = (5, 9, 9)
    motion_ext_chnls: tuple = (256,)
    motion_int_chnls: tuple = (256, 256, 512)
    motion_corr_func: str = "cosine"
    motion_n_encoders: int = 1
    motion_use_layerscale: bool = False
    motion_layerscale_init: float = 1e-5
    motion_use_layernorm: bool = False
    motion_use_syncbn: bool = False
    motion_injection_point: str = "vision_encoder"  # "vision_encoder" or "vl_input"
    motion_pool_type: str = "avg"  # "avg" or "conv" (spatial pooling for vl_input)
    motion_drop: bool = True  # drop motion module tokens at internal_projection layer
    motion_gradient_check: bool = False  # log motion module gradient norms during training
    motion_int_mode: str = "lite"  # "lite" (1x1 Conv3d L-fuse, default) or "full" (3-layer 3x3 conv stack)

    # Physics (tactile/torque) configuration (phase-2 add-on)
    use_physics: bool = False
    physics_keys: list[str] = field(default_factory=list)  # e.g., ["tactile", "torque"]
    physics_dims: list[int] = field(
        default_factory=list,
    )  # Per-key dimensions, aligned with `physics_keys` (e.g., [30, 7])
    physics_loss_weight: float = 0.1
    allow_missing_physics: bool = False  # If True, samples without physics data are zero-filled and attention-masked
    physics_delta_indices: list[int] | None = None  # Injected from modality_configs in setup.py; d<=0 = hist, d>0 = fut
    physics_use_flow_matching: bool = True  # False switches to the all-conditioning + MSE loss path
    physics_dropout_prob: float = 0.0
    """Per-sample dropout on physics conditioning tokens during training.
    Flow-matching mode drops only history tokens; the MSE-loss path drops
    the full sequence."""

    @property
    def physics_dim(self) -> int:
        """Total physics dimension, derived from physics_dims."""
        return sum(self.physics_dims) if self.physics_dims else 0

    # Optimizer & scheduler
    optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    scheduler_decay_lr: float = 1e-5

    # Precision & compute
    use_bf16: bool = True
    # Top-level flag; ``diffusion_model_cfg["gradient_checkpointing"]`` above
    # remains the one MSAT actually constructs from.
    gradient_checkpointing: bool = False

    attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa"
