# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for SmolVLA model.

This module provides dataclass configurations for the SmolVLA flow matching
vision-language-action model.
For CLI usage, use the YAML config in `configs/getiaction/smolvla.yaml`:
    getiaction fit --config configs/getiaction/smolvla.yaml
The YAML config is set up for minimum hardware (~8GB VRAM) with clear
comments on how to adjust for different GPU sizes.
Example (API):
    >>> from getiaction.policies.smolvla import SmolVLAConfig
    >>> config = SmolVLAConfig(
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass

from getiaction.config import Config


@dataclass
class SmolVLAConfig(Config):
    """Configuration for SmolVLA flow matching model."""

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = (
        False  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights
    )

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same
    # number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Ensures that the number of action steps does not exceed the chunk size,
        as the chunk size represents the upper bound for action steps per model invocation.

        Raises:
            ValueError: If n_action_steps is greater than chunk_size.
        """
        if self.n_action_steps > self.chunk_size:
            msg = (
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(msg)
