# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Cosmos3 policy.

This module provides dataclass configuration for the NVIDIA Cosmos 3
diffusion/flow-matching world model policy.

Example (CLI):
    physicalai fit --config configs/physicalai/cosmos3/pusht/default.yaml

Example (API):
    >>> from physicalai.policies.cosmos3 import Cosmos3Config
    >>> config = Cosmos3Config(embodiment="pusht")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.config import Config

_DEFAULT_PEFT_HEAD_MULT = 2.0
_DEFAULT_FULL_HEAD_MULT = 10.0
_EPS = 1e-6
DEFAULT_COSMOS3_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"


@dataclass(frozen=True)
class Cosmos3Config(Config):
    """Configuration for Cosmos 3 policy.

    Attributes:
        embodiment: Required embodiment identifier ("pusht", "droid_lerobot", or "aloha").
            Mapped internally to a numeric domain id, an action space, a normalization
            method and gripper handling.
        pretrained_model_name_or_path: Hugging Face repo ID or local checkpoint path.
            Defaults to "nvidia/Cosmos3-Edge".
        revision: Pinned git commit SHA for model and checkpoint downloads (lib.security rule 9).
            Defaults to DEFAULT_COSMOS3_REVISION.
        mode: Training mode. "peft" trains LoRA/DoRA on attention projections and
            a domain action head; "full" fine-tunes the generation tower and vision
            projections. Defaults to "peft".
        lora_enabled: Whether LoRA/DoRA fine-tuning is enabled. True implies mode="peft",
            False implies mode="full". Defaults to True.
        paradigm: Denoising objective. Options: "policy" (state + frame -> actions + video),
            "fd" (frame + actions -> video), "id" (video -> actions), or "joint" (random mix).
            Defaults to "policy".
        lora_rank: LoRA/DoRA rank (dimension of the low-rank decomposition). Defaults to 32.
        lora_alpha: LoRA scaling numerator (scaling = lora_alpha / lora_rank). Defaults to
            None, which resolves to lora_rank (i.e. scaling of 1.0).
        lora_dropout: Dropout probability applied to LoRA adapter inputs. Defaults to 0.05.
        lora_use_dora: Whether to use Weight-Decomposed Low-Rank Adaptation (DoRA). Defaults to False.
        head_lr_mult: Multiplier on optimizer_lr for the domain action head.
            Defaults to 2.0 for "peft" and 10.0 for "full".
        action_weight: Weight of the action loss vs video loss in flow matching. Defaults to 10.0.
        chunk_size: Size of the action chunk predicted. Defaults to 32.
        n_action_steps: Number of action steps to execute per replanning step. Defaults to 32.
        resolution_tier: Short-side resolution in pixels for video conditioning.
            Must be one of (256, 480, 720). Defaults to 256.
        fps: Frames per second for conditioning and generation. Defaults to 10.
        gradient_checkpointing: Enable gradient checkpointing for memory optimization. Defaults to True.
        action_space: Optional override of the embodiment's action space ("identity" or
            "joint_pos"). When None, resolved automatically from the embodiment
            (droid_lerobot -> joint_pos, others -> identity). Defaults to None.
        view_point: Optional camera viewpoint tag (e.g., "concat_view", "top_down_2d_view").
            When None, inferred from the embodiment and composition. Defaults to None.
        normalizer_stats_path: Optional path to a cosmos-format action-normalizer stats JSON
            (flat ``{"q01", "q99"}`` or nested ``{"global", "global_raw"}``). When set, the
            embodiment's normalization method is resolved against these stats (asserting the
            stats width matches the raw action dim); required to run a pre-trained per-embodiment
            head (e.g. a released Cosmos policy) that expects quantile-normalized actions. When
            None, identity embodiments derive the affine from the dataset's raw-column stats,
            while joint_pos (DROID) keeps actions un-normalized. Defaults to None.
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
        rank: Deprecated alias for `lora_rank`.
        alpha_scale: Deprecated alias for scaling factor (resolves lora_alpha = round(alpha_scale * lora_rank)).
        dora: Deprecated alias for `lora_use_dora`.
        grad_checkpoint: Deprecated alias for `gradient_checkpointing`.
        pretrained_name_or_path: Alias for `pretrained_model_name_or_path`.
    """

    embodiment: str
    pretrained_model_name_or_path: str = "nvidia/Cosmos3-Edge"
    revision: str | None = DEFAULT_COSMOS3_REVISION
    mode: Literal["peft", "full"] = "peft"
    lora_enabled: bool = True
    paradigm: Literal["policy", "fd", "id", "joint"] = "policy"
    lora_rank: int = 32
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    lora_use_dora: bool = False
    head_lr_mult: float = 2.0
    action_weight: float = 10.0
    chunk_size: int = 32
    n_action_steps: int | None = None
    resolution_tier: int = 256
    fps: int = 10
    gradient_checkpointing: bool = True
    action_space: str | None = None
    view_point: str | None = None
    normalizer_stats_path: str | None = None
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

    # Backwards-compatibility aliases
    rank: int | None = None
    alpha_scale: float | None = None
    dora: bool | None = None
    grad_checkpoint: bool | None = None
    pretrained_name_or_path: str | None = None

    def _set_frozen(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)  # ruff: ignore[unnecessary-dunder-call]

    def _resolve_aliases(self) -> None:
        """Resolve legacy hyperparameter aliases."""
        if self.pretrained_name_or_path is not None:
            self._set_frozen("pretrained_model_name_or_path", self.pretrained_name_or_path)

        if self.rank is not None:
            self._set_frozen("lora_rank", self.rank)

        if self.alpha_scale is not None and self.lora_alpha is None:
            self._set_frozen("lora_alpha", round(self.alpha_scale * self.lora_rank))

        if self.lora_alpha is None:
            self._set_frozen("lora_alpha", self.lora_rank)

        if self.dora is not None:
            self._set_frozen("lora_use_dora", self.dora)

        if self.grad_checkpoint is not None:
            self._set_frozen("gradient_checkpointing", self.grad_checkpoint)

        self._set_frozen("rank", self.lora_rank)
        self._set_frozen(
            "alpha_scale",
            float(self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0,
        )
        self._set_frozen("dora", self.lora_use_dora)
        self._set_frozen("grad_checkpoint", self.gradient_checkpointing)

    def _sync_mode_and_lora(self) -> None:
        """Synchronize training mode and lora_enabled flag."""
        if self.mode == "full":
            self._set_frozen("lora_enabled", value=False)
        elif not self.lora_enabled:
            self._set_frozen("mode", "full")
        else:
            self._set_frozen("lora_enabled", value=True)
            self._set_frozen("mode", "peft")

    def _validate(self) -> None:
        """Validate parameter ranges and types.

        Raises:
            ValueError: If any configuration parameter has an invalid value.
        """
        if self.mode not in {"peft", "full"}:
            msg = f"Invalid mode: {self.mode}. Must be 'peft' or 'full'."
            raise ValueError(msg)

        if self.paradigm not in {"policy", "fd", "id", "joint"}:
            msg = f"Invalid paradigm: {self.paradigm}. Must be 'policy', 'fd', 'id', or 'joint'."
            raise ValueError(msg)

        if self.lora_rank <= 0:
            msg = f"lora_rank must be positive, got {self.lora_rank}."
            raise ValueError(msg)

        if self.lora_alpha is not None and self.lora_alpha <= 0:
            msg = f"lora_alpha must be positive, got {self.lora_alpha}."
            raise ValueError(msg)

        if not 0.0 <= self.lora_dropout < 1.0:
            msg = f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}."
            raise ValueError(msg)

        if self.n_action_steps is None:
            self._set_frozen("n_action_steps", self.chunk_size)
        elif self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            raise ValueError(msg)

        if self.resolution_tier not in {256, 480, 720}:
            msg = f"Invalid resolution_tier: {self.resolution_tier}. Must be one of (256, 480, 720)."
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float32", "float16"}:
            msg = f"Invalid dtype: {self.dtype}. Must be 'bfloat16', 'float32', or 'float16'."
            raise ValueError(msg)

        if self.mode == "full" and abs(self.head_lr_mult - _DEFAULT_PEFT_HEAD_MULT) < _EPS:
            self._set_frozen("head_lr_mult", _DEFAULT_FULL_HEAD_MULT)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._resolve_aliases()
        self._validate()
        self._sync_mode_and_lora()
