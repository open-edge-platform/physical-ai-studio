# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the XR1 model.

This module provides the dataclass configuration for the XR1 (Xiaomi-Robotics-1)
flow-matching vision-language-action model: a Qwen3-VL backbone paired with a DiT
action expert in a Mixture-of-Transformers layout.

Example (CLI):
    physicalai fit --config configs/physicalai/xr1.yaml

Example (API):
    >>> from physicalai.policies.xr1 import XR1Config
    >>> config = XR1Config(dit_num_layers=4, dit_hidden_size=256)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from physicalai.config import Config

from physicalai.data import (
    Feature,  # noqa: TC001 - needed at runtime for type hint resolution
)

MIN_CHOICES = 2
IMAGE_RESOLUTION_RANK = 2


@dataclass(frozen=True)
class XR1Config(Config):
    """Configuration for the XR1 flow-matching model.

    Defaults follow the reference implementation unless noted. Three defaults
    deliberately differ, because the upstream value is either not portable or not
    supervisable by LeRobot datasets: ``vlm_attn_implementation``, ``async_train``
    and ``enable_choice_head``.

    Attributes:
        vlm_model_id: HuggingFace id of the Qwen3-VL backbone. Defaults to
            ``"Qwen/Qwen3-VL-4B-Instruct"``.
        vlm_pretrained: Load pretrained backbone weights from ``vlm_model_id``.
            Defaults to True. Set False to build a randomly initialized backbone
            from its config, which is what unit tests and smoke runs use to avoid a
            multi-gigabyte download.
        vlm_config_overrides: Optional nested overrides applied to the backbone
            config, e.g. ``{"text_config": {"num_hidden_layers": 4}}``. Only useful
            with ``vlm_pretrained=False``; the DiT geometry must stay compatible
            with the resulting backbone.
        vlm_attn_implementation: Attention backend for the VLM. Defaults to
            ``"sdpa"``. The reference implementation hardcodes
            ``"flash_attention_2"``, which is not a library dependency and cannot
            be traced for export.
        dtype: Precision for model weights. Options: ``"bfloat16"``,
            ``"float32"``. Defaults to ``"bfloat16"``.
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Number of action steps to predict (action horizon). Defaults
            to 30.
        n_action_steps: Number of action steps to execute per inference call.
            Defaults to 30.
        max_state_dim: State vector dimension; shorter vectors are padded.
            Defaults to 32. The reference implementation fixes this at 60 for its
            dual-arm layout; making it a config value lets lower-dimensional
            datasets train without touching the architecture.
        max_action_dim: Action vector dimension; shorter vectors are padded.
            Defaults to 32.
        state_len: Number of state tokens in the DiT query sequence. Defaults
            to 1.
        dit_num_layers: Number of DiT decoder layers. Defaults to 36. Must not
            exceed the VLM's layer count, because each DiT layer attends over the
            corresponding VLM cache layer.
        dit_hidden_size: DiT hidden width. Defaults to 1024.
        dit_head_dim: DiT attention head dim. Must match the VLM head dim.
            Defaults to 128.
        dit_kv_heads: DiT key/value heads. Must match the VLM kv heads. Defaults
            to 8.
        num_inference_steps: Euler integration steps for flow inference. Defaults
            to 5.
        flow_sampling: Training timestep distribution. Options: ``"beta"``,
            ``"logit_normal"``, ``"uniform"``. Defaults to ``"beta"``.
        beta_alpha: Alpha of the Beta timestep prior. Defaults to 1.5.
        beta_beta: Beta of the Beta timestep prior. Defaults to 1.0.
        training_repeat: Per-sample training repeat factor; each sample is
            denoised at several timesteps per step. Defaults to 4.
        prefix_mask_prob: Probability of masking a prefix action token during
            training. Defaults to 0.5.
        async_train: Randomly condition on an action prefix during training.
            Defaults to False; the reference implementation defaults to True, but
            the prefix path expects data carrying an executed action prefix.
        enable_freq: Add the frequency-domain loss term. Defaults to True.
        freq_coefficient: Weight of the frequency-domain loss term. Defaults
            to 1.0.
        freq_excluded_dims: Action dimensions excluded from the frequency loss
            (gripper-like dimensions in the reference layout). Defaults to
            ``(17, 18, 19)``.
        enable_choice_head: Train the discrete action-choice head. Defaults to
            False. The head needs per-sample choice targets, dedicated prompt
            tokens and a state-embedding input that only the reference
            implementation's vendored VLM accepts; it is training-only, so
            inference is unaffected when disabled.
        n_choices: Number of action candidates produced by the choice head.
            Defaults to 5.
        image_resolution: Target image resolution (height, width). Defaults to
            (256, 256).
        camera_views: Ordered camera view names embedded into the prompt.
            Defaults to ``("base", "wrist_left")``.
        tokenizer_max_length: Maximum length for tokenizer output. Defaults
            to 256.
        gradient_checkpointing: Enable gradient checkpointing. Defaults to True.
        freeze_vlm: Freeze the whole VLM and train only the DiT action expert and
            projectors. Defaults to False. This is the recipe that fits a single
            24 GB GPU.
        freeze_vision_encoder: Freeze only the vision tower. Defaults to False.
        normalization_mode: Normalization for state/action features.
            ``"QUANTILES"`` maps to [-1, 1] using the 1st and 99th percentiles;
            ``"MEAN_STD"`` uses zero-mean unit-variance. Defaults to
            ``"MEAN_STD"``.
        optimizer_lr: Learning rate. Defaults to 1e-4.
        optimizer_betas: Adam beta coefficients. Defaults to (0.9, 0.95).
        optimizer_eps: Adam epsilon. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 0.1.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 1.0.
        scheduler_warmup_steps: Warmup steps. Defaults to 2000.
        scheduler_decay_steps: Cosine decay steps; ``None`` uses the total
            training steps. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 5e-7.
        input_features: Optional explicit observation feature schema. When
            ``None`` it is traced from the training dataset in :meth:`XR1.setup`.
            Must be provided together with ``output_features``.
        output_features: Optional explicit action feature schema. Must be
            provided together with ``input_features``.
    """

    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_pretrained: bool = True
    vlm_config_overrides: dict[str, Any] | None = None
    vlm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    max_state_dim: int = 32
    max_action_dim: int = 32
    state_len: int = 1

    dit_num_layers: int = 36
    dit_hidden_size: int = 1024
    dit_head_dim: int = 128
    dit_kv_heads: int = 8

    num_inference_steps: int = 5
    flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta"
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    training_repeat: int = 4
    prefix_mask_prob: float = 0.5
    async_train: bool = False

    enable_freq: bool = True
    freq_coefficient: float = 1.0
    freq_excluded_dims: tuple[int, ...] = (17, 18, 19)

    enable_choice_head: bool = False
    n_choices: int = 5

    image_resolution: tuple[int, int] = (256, 256)
    camera_views: tuple[str, ...] = field(default_factory=lambda: ("base", "wrist_left"))
    tokenizer_max_length: int = 256

    gradient_checkpointing: bool = True
    freeze_vlm: bool = False
    freeze_vision_encoder: bool = False

    normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "MEAN_STD"

    optimizer_lr: float = 1.0e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 2_000
    scheduler_decay_steps: int | None = 30_000
    scheduler_decay_lr: float = 5.0e-7

    input_features: list[Feature] | None = None
    output_features: list[Feature] | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Delegates to focused validators. Checks that can only be made against the
        VLM's own config (the DiT depth and KV geometry must match the backbone)
        are enforced when the model is built, in
        :class:`physicalai.policies.xr1.vla.XR1Model`.
        """
        self._validate_structure()
        self._validate_dit_geometry()
        self._validate_training_options()

    def _validate_structure(self) -> None:
        """Validate observation and action shapes.

        Raises:
            ValueError: If a structural size is inconsistent.
        """
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)

        if self.chunk_size < 1:
            msg = f"chunk_size ({self.chunk_size}) must be positive"
            raise ValueError(msg)

        if self.state_len < 1:
            msg = f"state_len ({self.state_len}) must be positive"
            raise ValueError(msg)

    def _validate_dit_geometry(self) -> None:
        """Validate the action expert's head geometry.

        Raises:
            ValueError: If the DiT head layout is inconsistent.
        """
        if self.dit_hidden_size % self.dit_head_dim != 0:
            msg = f"dit_hidden_size ({self.dit_hidden_size}) must be divisible by dit_head_dim ({self.dit_head_dim})"
            raise ValueError(msg)

        num_heads = self.dit_hidden_size // self.dit_head_dim
        if num_heads < self.dit_kv_heads:
            msg = f"DiT num_heads ({num_heads}) must be >= dit_kv_heads ({self.dit_kv_heads})"
            raise ValueError(msg)

        if num_heads % self.dit_kv_heads != 0:
            msg = f"DiT num_heads ({num_heads}) must be divisible by dit_kv_heads ({self.dit_kv_heads})"
            raise ValueError(msg)

        if self.dit_num_layers < 1:
            msg = f"dit_num_layers ({self.dit_num_layers}) must be positive"
            raise ValueError(msg)

    def _validate_training_options(self) -> None:
        """Validate loss, sampling and feature options.

        Raises:
            ValueError: If a training option is inconsistent.
        """
        if self.num_inference_steps < 1:
            msg = f"num_inference_steps ({self.num_inference_steps}) must be positive"
            raise ValueError(msg)

        if self.training_repeat < 1:
            msg = f"training_repeat ({self.training_repeat}) must be positive"
            raise ValueError(msg)

        if not 0.0 <= self.prefix_mask_prob <= 1.0:
            msg = f"prefix_mask_prob ({self.prefix_mask_prob}) must be in [0, 1]"
            raise ValueError(msg)

        if self.enable_choice_head and self.n_choices < MIN_CHOICES:
            msg = f"n_choices ({self.n_choices}) must be at least 2 when the choice head is enabled"
            raise ValueError(msg)

        if any(dim < 0 for dim in self.freq_excluded_dims):
            msg = f"freq_excluded_dims must be non-negative, got {self.freq_excluded_dims}"
            raise ValueError(msg)

        if not self.camera_views:
            msg = "camera_views must contain at least one view"
            raise ValueError(msg)

        if len(set(self.camera_views)) != len(self.camera_views):
            msg = f"camera_views must be unique, got {self.camera_views}"
            raise ValueError(msg)

        if len(self.image_resolution) != IMAGE_RESOLUTION_RANK or any(size < 1 for size in self.image_resolution):
            msg = f"image_resolution must be two positive integers, got {self.image_resolution}"
            raise ValueError(msg)

        if self.vlm_config_overrides and self.vlm_pretrained:
            msg = "vlm_config_overrides only applies when vlm_pretrained=False"
            raise ValueError(msg)

        if (self.input_features is None) != (self.output_features is None):
            msg = "input_features and output_features must be provided together"
            raise ValueError(msg)
