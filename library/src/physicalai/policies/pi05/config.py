# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Pi05 model.

This module provides dataclass configurations for the Pi05 flow matching
vision-language-action model.

Example (CLI):
    physicalai fit --config configs/physicalai/pi05.yaml

Example (API):
    >>> from physicalai.policies.pi05 import Pi05Config
    >>> config = Pi05Config()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.config import Config


@dataclass(frozen=True)
class Pi05Config(Config):
    """Configuration for Pi05 flow matching model.

    Attributes:
        paligemma_variant: Gemma variant for the VLM backbone. Defaults to "gemma_2b".
        action_expert_variant: Gemma variant for the action expert. Defaults to "gemma_300m".
        dtype: Precision for model weights. Options: "bfloat16", "float32". Defaults to "float32".
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Number of action steps to predict. Defaults to 50.
        n_action_steps: Number of action steps to execute. Defaults to 50.
        max_state_dim: Maximum dimension for state vectors; shorter vectors will be padded. Defaults to 32.
        max_action_dim: Maximum dimension for action vectors; shorter vectors will be padded. Defaults to 32.
        num_inference_steps: Number of decoding steps for flow matching. Defaults to 10.
        time_sampling_beta_alpha: Alpha parameter for beta distribution time sampling. Defaults to 1.5.
        time_sampling_beta_beta: Beta parameter for beta distribution time sampling. Defaults to 1.0.
        time_sampling_scale: Scale factor for time sampling. Defaults to 0.999.
        time_sampling_offset: Offset for time sampling. Defaults to 0.001.
        min_period: Minimum period for sine-cosine positional encoding. Defaults to 4e-3.
        max_period: Maximum period for sine-cosine positional encoding. Defaults to 4.0.
        image_resolution: Target image resolution (height, width). Defaults to (224, 224).
        empty_cameras: Number of empty camera slots to add. Defaults to 0.
        tokenizer_max_length: Maximum length for tokenizer output. Defaults to 200.
        gradient_checkpointing: Enable gradient checkpointing for memory optimization. Defaults to True.
        compile_model: Whether to use torch.compile. Defaults to False.
        compile_mode: Torch compile mode. Defaults to "max-autotune".
        freeze_vision_encoder: Whether to freeze vision encoder during training. Defaults to False.
        train_expert_only: Whether to train only the action expert. Defaults to True.
        optimizer_lr: Learning rate for the optimizer. Defaults to 2.5e-5.
        optimizer_betas: Beta coefficients for Adam optimizer. Defaults to (0.9, 0.95).
        optimizer_eps: Epsilon for optimizer numerical stability. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 0.01.
        optimizer_grad_clip_norm: Maximum gradient norm for clipping. Defaults to 1.0.
        scheduler_warmup_steps: Number of warmup steps. Defaults to 1000.
        scheduler_decay_steps: Number of cosine decay steps. When ``None``,
            automatically set to the total training steps via
            ``trainer.estimated_stepping_batches``. Defaults to None.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 2.5e-6.
        use_random_input_noise: Whether to use random noise as the initial input for the denoising process
            during inference. If False, zeros are used instead. Defaults to False.
    """

    paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b"
    action_expert_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_300m"
    dtype: Literal["bfloat16", "float32"] = "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)

    empty_cameras: int = 0

    tokenizer_max_length: int = 200

    gradient_checkpointing: bool = True
    compile_model: bool = False
    compile_mode: str = "default"

    freeze_vision_encoder: bool = False
    train_expert_only: bool = True

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int | None = None
    scheduler_decay_lr: float = 2.5e-6

    use_random_input_noise: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)

        if self.paligemma_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid paligemma_variant: {self.paligemma_variant}"
            raise ValueError(msg)

        if self.action_expert_variant not in {"gemma_300m", "gemma_2b"}:
            msg = f"Invalid action_expert_variant: {self.action_expert_variant}"
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)
