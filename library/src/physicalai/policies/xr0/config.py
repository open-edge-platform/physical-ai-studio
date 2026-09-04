# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the XR0 model.

This module provides the dataclass configuration for the XR0 flow-matching
vision-language-action model (Qwen3-VL-4B backbone + DiT action expert).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from physicalai.config import Config

from physicalai.data import Feature  # noqa: TC001 - Needed at runtime for type hint resolution


@dataclass(frozen=True)
class XR0Config(Config):
    """Configuration for the XR0 flow-matching model.

    Attributes:
        vlm_model_id: HuggingFace id of the Qwen3-VL backbone. Defaults to
            ``"Qwen/Qwen3-VL-4B-Instruct"``.
        vlm_attn_implementation: Attention backend for the VLM. Defaults to
            ``"flash_attention_2"``.
        dtype: Precision for model weights. Options: ``"bfloat16"``,
            ``"float16"``, ``"float32"``. Defaults to ``"bfloat16"``. Use
            ``"float16"`` when exporting to OpenVINO for Intel GPU: the GPU
            plugin cannot build the OpenCL kernel for bf16 attention permutes,
            while the f16 kernel builds correctly.
        n_obs_steps: Number of observation steps to use. Defaults to 1. Unused:
            XR0 always conditions on the single current observation; kept only
            for config parity with other policies.
        chunk_size: Number of action steps to predict (action horizon).
            Defaults to 30.
        n_action_steps: Number of action steps to execute. Defaults to 30.
        max_state_dim: State vector dimension; shorter vectors are padded.
            Defaults to 32.
        max_action_dim: Action vector dimension; shorter vectors are padded.
            Defaults to 32.
        state_len: Number of state tokens. Defaults to 1.
        dit_num_layers: Number of DiT decoder layers. Defaults to 16.
        dit_hidden_size: DiT hidden width. Defaults to 1024.
        dit_head_dim: DiT attention head dim (matches the VLM head dim).
            Defaults to 128.
        dit_kv_heads: DiT key/value heads (matches the VLM kv heads). Defaults
            to 8.
        num_inference_steps: Euler integration steps for flow inference.
            Defaults to 5.
        flow_sampling: Training timestep distribution. Options: ``"beta"``,
            ``"logit_normal"``, ``"uniform"``. Defaults to ``"beta"``.
        local_window: Local-attention window for the action tokens. Defaults
            to 4.
        training_repeat: Per-sample training repeat factor. Defaults to 4.
        enable_freq: Add the frequency-domain loss term. Defaults to True.
        prefix_mask_prob: Probability of masking a prefix token during training.
            Defaults to 0.5.
        async_train: Randomly condition on an action prefix during training.
            Defaults to False.
        image_resolution: Target image resolution (height, width). Defaults to
            (256, 256).
        camera_views: Ordered camera view names embedded into the prompt.
            Defaults to ("base", "wrist_left").
        tokenizer_max_length: Maximum length for tokenizer output. Defaults to
            256.
        gradient_checkpointing: Enable gradient checkpointing for memory
            optimization. Defaults to True.
        compile_model: Whether to use torch.compile. Defaults to False.
        compile_mode: Torch compile mode. Defaults to ``"max-autotune"``.
        freeze_vision_encoder: Whether to freeze the vision encoder during
            training. Defaults to False.
        freeze_input_embeddings: Whether to freeze the VLM token-embedding table
            during training. Matches the original XR0 recipe and saves the
            embedding's gradients / optimizer state. Defaults to True.
        normalize_state: Whether to normalize the proprioceptive state with the
            dataset's per-dimension mean/std before feeding it to the model.
            Defaults to False, which preserves the raw-state behavior of the
            upstream XR0 recipe and keeps existing checkpoints/exports byte
            compatible. Enable it for embodiments whose raw state is off the
            scale the pretrained checkpoint expects (e.g. joint positions in
            degrees). The resulting mean/std become part of the trained
            checkpoint's contract and are baked into the exported manifest.
        action_mode: How the action target is represented. ``"absolute"`` (the
            default) predicts the raw action directly. ``"delta"`` predicts the
            per-step delta relative to the current state (``action[t] - state``),
            matching the pretrained XR0 flow head's delta prior; the inverse
            (``delta + state``) is applied at inference. Delta mode requires
            per-timestep delta stats supplied via ``action_delta_mean`` /
            ``action_delta_std``.
        normalization_mode: Normalization method for state/action features.
            ``"QUANTILES"`` maps data to [-1, 1] using the 1st and 99th
            percentiles; ``"MEAN_STD"`` uses zero-mean unit-variance
            normalization. Defaults to ``"QUANTILES"``.
        optimizer_lr: Learning rate for the optimizer. Defaults to 1e-4.
        optimizer_betas: Beta coefficients for Adam optimizer. Defaults to
            (0.9, 0.95).
        optimizer_eps: Epsilon for optimizer numerical stability. Defaults to
            1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 0.1.
        optimizer_grad_clip_norm: Maximum gradient norm for clipping. Defaults
            to 1.0.
        scheduler_warmup_steps: Number of warmup steps. Defaults to 2000.
        scheduler_decay_steps: Number of cosine decay steps. When ``None``,
            automatically set to the total training steps. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 5e-7.
        input_features: Optional explicit observation feature schema
            (``list[Feature]``). When ``None`` it is traced back from the
            training dataset in :meth:`XR0.setup`. Must be provided together
            with ``output_features``.
        output_features: Optional explicit action feature schema
            (``list[Feature]``). Must be provided together with
            ``input_features``.
    """

    vlm_model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    max_state_dim: int = 32
    max_action_dim: int = 32
    state_len: int = 1

    dit_num_layers: int = 16
    dit_hidden_size: int = 1024
    dit_head_dim: int = 128
    dit_kv_heads: int = 8

    num_inference_steps: int = 5
    flow_sampling: Literal["beta", "logit_normal", "uniform"] = "beta"
    local_window: int = 4
    training_repeat: int = 4
    enable_freq: bool = True
    prefix_mask_prob: float = 0.5
    async_train: bool = False

    image_resolution: tuple[int, int] = (256, 256)
    camera_views: tuple[str, ...] = field(default_factory=lambda: ("base", "wrist_left"))
    tokenizer_max_length: int = 256

    gradient_checkpointing: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    freeze_vision_encoder: bool = False
    freeze_input_embeddings: bool = True
    normalize_state: bool = False

    action_mode: Literal["absolute", "delta"] = "absolute"

    normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES"

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

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        if self.n_action_steps > self.chunk_size:
            msg = f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)

        if self.dit_hidden_size % self.dit_head_dim != 0:
            msg = f"dit_hidden_size ({self.dit_hidden_size}) must be divisible by dit_head_dim ({self.dit_head_dim})"
            raise ValueError(msg)

        num_heads = self.dit_hidden_size // self.dit_head_dim
        if num_heads < self.dit_kv_heads:
            msg = f"DiT num_heads ({num_heads}) must be >= dit_kv_heads ({self.dit_kv_heads})"
            raise ValueError(msg)

        if not self.camera_views:
            msg = "camera_views must contain at least one view"
            raise ValueError(msg)
