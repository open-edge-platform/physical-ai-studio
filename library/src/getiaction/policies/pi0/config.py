# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Pi0/Pi0.5 models.

This module provides dataclass configurations for the Pi0 and Pi0.5 flow matching
vision-language-action models.

The configuration supports both Pi0 and Pi0.5 variants via the `variant` field:
- Pi0: Uses continuous state input with MLP timestep injection
- Pi0.5: Uses discrete state input (tokenized) with adaRMSNorm timestep injection

For CLI usage, use the YAML config in `configs/pi0/pi0.yaml`:

    getiaction fit --config configs/pi0/pi0.yaml

The YAML config is set up for minimum hardware (~8GB VRAM) with clear
comments on how to adjust for different GPU sizes.

Example (API):
    >>> from getiaction.policies.pi0 import Pi0Config
    >>> config = Pi0Config(
    ...     variant="pi0",
    ...     action_dim=14,
    ...     action_horizon=50,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from getiaction.config import Config


@dataclass
class Pi0Config(Config):
    """Configuration for Pi0/Pi0.5 flow matching model.

    The same config supports both Pi0 and Pi0.5 via the `variant` field.
    Pi0.5 uses discrete state input and adaRMSNorm for timestep injection.

    Attributes:
        variant: Model variant - "pi0" or "pi05".
        paligemma_variant: PaliGemma backbone size ("gemma_300m" or "gemma_2b").
        action_expert_variant: Action expert size ("gemma_300m" or "gemma_2b").
        dtype: Compute dtype ("bfloat16" or "float32").
        action_dim: Action dimension (will be padded to max_action_dim).
        action_horizon: Number of action steps to predict (chunk_size).
        max_state_dim: Maximum state dimension for padding.
        max_action_dim: Maximum action dimension for padding.
        max_token_len: Maximum tokenized prompt length. Auto-set if None.
        num_inference_steps: Number of denoising steps during inference.
        image_resolution: Input image resolution (height, width).
        tune_paligemma: Whether to fine-tune PaliGemma backbone.
        tune_action_expert: Whether to fine-tune action expert.
        lora_rank: LoRA rank. 0 disables LoRA.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: Which modules to apply LoRA to.

    Examples:
        Basic Pi0 config:

        >>> config = Pi0Config(action_dim=14)
        >>> print(config.is_pi05)
        False

        Pi0.5 with LoRA:

        >>> config = Pi0Config(
        ...     variant="pi05",
        ...     lora_rank=16,
        ...     lora_alpha=32,
        ... )
    """

    # Model variant
    variant: Literal["pi0", "pi05"] = "pi0"

    # Architecture - backbone
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"

    # Dimensions
    action_dim: int = 32
    action_horizon: int = 50  # Also called chunk_size
    max_state_dim: int = 32
    max_action_dim: int = 32
    max_token_len: int | None = None  # Auto: 200 for pi05, 48 for pi0

    # Inference
    num_inference_steps: int = 10

    # Flow matching parameters
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    time_scale: float = 0.999
    time_offset: float = 0.001
    time_min_period: float = 4e-3
    time_max_period: float = 4.0

    # Image processing
    image_resolution: tuple[int, int] = (224, 224)

    # Training - what to tune
    tune_paligemma: bool = False
    tune_action_expert: bool = True
    tune_vision_encoder: bool = False
    tune_projection_heads: bool = True  # Always train projection heads

    # LoRA configuration
    lora_rank: int = 0  # 0 = disabled
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "v_proj", "k_proj", "o_proj"),
    )

    # Gradient checkpointing for memory optimization
    gradient_checkpointing: bool = False

    # Optimizer/training hyperparameters
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    warmup_ratio: float = 0.05  # Warmup ratio (0.0-1.0) of total training steps
    grad_clip_norm: float = 1.0  # Gradient clipping norm (0.0 = disabled)

    def __post_init__(self) -> None:
        """Set defaults based on variant.

        Raises:
            ValueError: If variant or backbone variants are invalid.
        """
        if self.max_token_len is None:
            # Pi0.5 needs longer context for discrete state tokens
            self.max_token_len = 200 if self.variant == "pi05" else 48

        # Validate
        if self.variant not in {"pi0", "pi05"}:
            msg = f"variant must be 'pi0' or 'pi05', got '{self.variant}'"
            raise ValueError(msg)

        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"paligemma_variant must be 'gemma_300m' or 'gemma_2b', got '{self.paligemma_variant}'"
            raise ValueError(msg)

        if self.action_expert_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"action_expert_variant must be 'gemma_300m' or 'gemma_2b', got '{self.action_expert_variant}'"
            raise ValueError(msg)

    @property
    def is_pi05(self) -> bool:
        """Check if this is Pi0.5 variant."""
        return self.variant == "pi05"

    @property
    def use_discrete_state(self) -> bool:
        """Whether state is tokenized (Pi0.5) vs continuous (Pi0)."""
        return self.is_pi05

    @property
    def use_adarms(self) -> bool:
        """Whether to use adaRMSNorm for timestep conditioning (Pi0.5)."""
        return self.is_pi05

    @property
    def use_lora(self) -> bool:
        """Whether LoRA is enabled."""
        return self.lora_rank > 0
