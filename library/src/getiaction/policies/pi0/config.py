# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Pi0/Pi0.5 models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from getiaction.config import Config


@dataclass
class Pi0Config(Config):
    """Configuration for Pi0/Pi0.5 flow matching model."""

    variant: Literal["pi0", "pi05"] = "pi0"

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32
    max_token_len: int | None = None

    image_resolution: tuple[int, int] = (224, 224)

    num_inference_steps: int = 10

    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0
    time_scale: float = 0.999
    time_offset: float = 0.001
    time_min_period: float = 4e-3
    time_max_period: float = 4.0

    tune_paligemma: bool = False
    tune_action_expert: bool = False
    tune_vision_encoder: bool = False

    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "v_proj", "k_proj", "o_proj"),
    )

    gradient_checkpointing: bool = False

    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate and apply default values."""  # noqa: DOC501
        if self.max_token_len is None:
            self.max_token_len = 200 if self.variant == "pi05" else 48

        if self.variant not in {"pi0", "pi05"}:
            msg = f"variant must be 'pi0' or 'pi05', got '{self.variant}'"
            raise ValueError(msg)

        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"paligemma_variant must be 'gemma_300m' or 'gemma_2b', got '{self.paligemma_variant}'"
            raise ValueError(msg)

        if self.action_expert_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"action_expert_variant must be 'gemma_300m' or 'gemma_2b', got '{self.action_expert_variant}'"
            raise ValueError(msg)

        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps must be <= chunk_size. Got {self.n_action_steps} and {self.chunk_size}."
            raise ValueError(msg)

    @property
    def is_pi05(self) -> bool:
        """Return True if using Pi0.5 variant."""
        return self.variant == "pi05"

    @property
    def use_discrete_state(self) -> bool:
        """Return True if using discrete state encoding (Pi0.5)."""
        return self.is_pi05

    @property
    def use_adarms(self) -> bool:
        """Return True if using AdaRMSNorm (Pi0.5)."""
        return self.is_pi05

    @property
    def use_lora(self) -> bool:
        """Return True if LoRA is enabled."""
        return self.lora_rank > 0


@dataclass
class Pi05Config(Pi0Config):
    """Configuration for Pi0.5 flow matching model.

    Inherits from Pi0Config with Pi0.5-specific defaults:
    - variant: "pi05" (uses discrete state encoding and AdaRMS)
    - max_token_len: 200 (larger context window)
    """

    variant: Literal["pi0", "pi05"] = "pi05"
    max_token_len: int | None = 200
