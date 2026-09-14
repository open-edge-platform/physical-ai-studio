# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Cosmos3 policy.

This module provides dataclass configuration for the NVIDIA Cosmos 3
diffusion/flow-matching world model policy.

Example (CLI):
    physicalai fit --config configs/physicalai/cosmos3.yaml

Example (API):
    >>> from physicalai.policies.cosmos3 import Cosmos3Config
    >>> config = Cosmos3Config()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.config import Config

_DEFAULT_PEFT_HEAD_MULT = 2.0
_DEFAULT_FULL_HEAD_MULT = 10.0
_EPS = 1e-6


@dataclass(frozen=True)
class Cosmos3Config(Config):
    """Configuration for Cosmos 3 policy.

    Attributes:
        pretrained_model_name_or_path: Hugging Face repo ID or local checkpoint path.
            Defaults to "nvidia/Cosmos3-Edge".
        mode: Training mode. "peft" trains LoRA/DoRA on attention projections and
            a domain action head; "full" fine-tunes the generation tower and vision
            projections. Defaults to "peft".
        paradigm: Denoising objective. Options: "policy" (state + frame -> actions + video),
            "fd" (frame + actions -> video), "id" (video -> actions), or "joint" (random mix).
            Defaults to "policy".
        rank: LoRA/DoRA rank for PEFT mode. Defaults to 32.
        alpha_scale: LoRA alpha scaling factor (lora_alpha = round(alpha_scale * rank)).
            Defaults to 1.0.
        dora: Whether to use Weight-Decomposed Low-Rank Adaptation (DoRA). Defaults to False.
        head_lr_mult: Multiplier on optimizer_lr for the domain action head.
            Defaults to 2.0 for "peft" and 10.0 for "full".
        action_weight: Weight of the action loss vs video loss in flow matching. Defaults to 10.0.
        chunk_size: Size of the action chunk predicted. Defaults to 32.
        n_action_steps: Number of action steps to execute per replanning step. Defaults to 32.
        resolution_tier: Short-side resolution in pixels for video conditioning.
            Must be one of (256, 480, 720). Defaults to 256.
        fps: Frames per second for conditioning and generation. Defaults to 10.
        grad_checkpoint: Enable gradient checkpointing for memory optimization. Defaults to True.
        domain: Embodiment domain identifier (e.g., "pusht", "droid_lerobot", "bridge_orig_lerobot",
            "aloha", "libero"). Defaults to "pusht".
        prompt: Task instruction conditioning string. Defaults to "".
        guidance_scale: Classifier-free guidance scale for inference. Defaults to 3.0.
        flow_shift: Flow shift value for the UniPC multistep scheduler. Defaults to 8.0.
        num_inference_steps: Number of denoising steps during inference. Defaults to 4.
        dtype: Model precision for weights and computation.
            Options: "bfloat16", "float32", "float16". Defaults to "bfloat16".
        optimizer_lr: Base learning rate for AdamW optimizer. Defaults to 1e-4.
        optimizer_betas: Beta coefficients for AdamW optimizer. Defaults to (0.9, 0.999).
        optimizer_eps: Epsilon parameter for optimizer numerical stability. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient for AdamW optimizer. Defaults to 0.01.
        optimizer_grad_clip_norm: Maximum gradient norm for gradient clipping. Defaults to 1.0.
    """

    pretrained_model_name_or_path: str = "nvidia/Cosmos3-Edge"
    mode: Literal["peft", "full"] = "peft"
    paradigm: Literal["policy", "fd", "id", "joint"] = "policy"
    rank: int = 32
    alpha_scale: float = 1.0
    dora: bool = False
    head_lr_mult: float = 2.0
    action_weight: float = 10.0
    chunk_size: int = 32
    n_action_steps: int | None = None
    resolution_tier: int = 256
    fps: int = 10
    grad_checkpoint: bool = True
    domain: str = "pusht"
    prompt: str = ""
    guidance_scale: float = 3.0
    flow_shift: float = 8.0
    num_inference_steps: int = 4
    dtype: Literal["bfloat16", "float32", "float16"] = "bfloat16"

    # Optimizer hyperparameters
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter has an invalid value.
        """
        if self.mode not in {"peft", "full"}:
            msg = f"Invalid mode: {self.mode}. Must be 'peft' or 'full'."
            raise ValueError(msg)

        if self.paradigm not in {"policy", "fd", "id", "joint"}:
            msg = f"Invalid paradigm: {self.paradigm}. Must be 'policy', 'fd', 'id', or 'joint'."
            raise ValueError(msg)

        if self.n_action_steps is None:
            object.__setattr__(self, "n_action_steps", self.chunk_size)
        elif self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            raise ValueError(msg)

        if self.resolution_tier not in {256, 480, 720}:
            msg = f"Invalid resolution_tier: {self.resolution_tier}. Must be one of (256, 480, 720)."
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float32", "float16"}:
            msg = f"Invalid dtype: {self.dtype}. Must be 'bfloat16', 'float32', or 'float16'."
            raise ValueError(msg)

        # Set appropriate default head_lr_mult if user left it as default but switched to full mode
        if self.mode == "full" and abs(self.head_lr_mult - _DEFAULT_PEFT_HEAD_MULT) < _EPS:
            object.__setattr__(self, "head_lr_mult", _DEFAULT_FULL_HEAD_MULT)
