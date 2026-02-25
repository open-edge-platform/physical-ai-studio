# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for SmolVLA model.

This module provides dataclass configurations for the SmolVLA flow matching
vision-language-action model.
For CLI usage, use the YAML config in `configs/physicalai/smolvla.yaml`:
    physicalai fit --config configs/physicalai/smolvla.yaml
The YAML config is set up for minimum hardware (~8GB VRAM) with clear
comments on how to adjust for different GPU sizes.
Example (API):
    >>> from physicalai.policies.smolvla import SmolVLAConfig
    >>> config = SmolVLAConfig(
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass

from physicalai.config import Config


@dataclass(frozen=True)
class SmolVLAConfig(Config):
    """Configuration for SmolVLA flow matching model.

    Attributes:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Size of action chunks for prediction. Defaults to 50.
        n_action_steps: Number of action steps to execute per model invocation. Defaults to 50.
        max_state_dim: Maximum dimension for state vectors; shorter vectors will be padded. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors; shorter vectors will be padded. Defaults to 32.
        resize_imgs_with_padding: Target size (height, width) for image preprocessing with padding.
            Defaults to (512, 512).
        empty_cameras: Number of empty camera images to add. Used by smolvla_aloha_sim for adding empty wrist cameras.
            Defaults to 0.
        adapt_to_pi_aloha: Whether to convert joint and gripper values from standard Aloha space to pi internal
            runtime space. Defaults to False.
        tokenizer_max_length: Maximum length for tokenizer output. Defaults to 48.
        num_steps: Number of decoding steps for flow matching. Defaults to 10.
        use_cache: Whether to use attention caching for efficiency. Defaults to True.
        freeze_vision_encoder: Whether to freeze the vision encoder during fine-tuning. Defaults to True.
        train_expert_only: Whether to train only the action expert during fine-tuning. Defaults to True.
        train_state_proj: Whether to train the state projection layer. Defaults to True.
        optimizer_lr: Learning rate for the optimizer. Defaults to 1e-4.
        optimizer_betas: Beta coefficients for Adam optimizer. Defaults to (0.9, 0.95).
        optimizer_eps: Epsilon value for numerical stability in optimizer. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient for regularization. Defaults to 1e-10.
        optimizer_grad_clip_norm: Maximum gradient norm for gradient clipping. Defaults to 10.
        scheduler_warmup_steps: Number of warmup steps for learning rate scheduler. Defaults to 1000.
        scheduler_decay_steps: Number of decay steps for learning rate scheduler. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 2.5e-6.
        vlm_model_name: Name or path of the VLM backbone model to use.
            Defaults to "HuggingFaceTB/SmolVLM2-500M-Video-Instruct".
        load_vlm_weights: Whether to load pretrained VLM weights. Set to True when training expert from scratch.
            Defaults to False.
        add_image_special_tokens: Whether to add special tokens around image features. Defaults to False.
        attention_mode: Type of attention mechanism to use. Defaults to "cross_attn".
        prefix_length: Length of prefix for attention. Negative values indicate default behavior. Defaults to -1.
        pad_language_to: Padding strategy for language tokens ("longest" or "max_length"). Defaults to "longest".
        num_expert_layers: Number of layers in the action expert. Values <= 0 use same number as VLM. Defaults to -1.
        num_vlm_layers: Number of VLM layers to use (first N layers). Defaults to 16.
        self_attn_every_n_layers: Frequency of self-attention layer interleaving. Defaults to 2.
        expert_width_multiplier: Multiplier for action expert hidden size relative to VLM. Defaults to 0.75.
        min_period: Minimum period for sine-cosine positional encoding of timesteps. Defaults to 4e-3.
        max_period: Maximum period for sine-cosine positional encoding of timesteps. Defaults to 4.0.
    """

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    empty_cameras: int = 0

    adapt_to_pi_aloha: bool = False

    tokenizer_max_length: int = 48

    num_steps: int = 10

    use_cache: bool = True

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False

    add_image_special_tokens: bool = False

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    min_period: float = 4e-3
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
