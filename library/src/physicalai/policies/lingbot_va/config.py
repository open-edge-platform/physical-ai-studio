# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the LingBot-VA policy.

LingBot-VA is an autoregressive video-action world model built on the Wan2.2 video-diffusion
stack: a single dual-stream transformer interleaves the prediction of future video latents
and robot actions ("VA" = Video-Action).

Defaults match the upstream LIBERO configuration and the released checkpoints
(``lerobot/lingbot_va_libero_long``).

Example (CLI):
    physicalai fit --config configs/physicalai/lingbot_va.yaml

Example (API):
    >>> from physicalai.policies.lingbot_va import LingBotVAConfig
    >>> config = LingBotVAConfig()
    >>> config.chunk_size
    16
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.config import Config

VAE_TEMPORAL_DOWNSAMPLE = 4
"""Temporal downsample factor of the Wan2.2 VAE (4 observed frames -> 1 latent frame)."""

VAE_SPATIAL_DOWNSAMPLE = 16
"""Combined spatial downsample of the VAE encoder and the transformer's latent patching."""


@dataclass(frozen=True)
class LingBotVAConfig(Config):
    """Configuration for the LingBot-VA autoregressive video-action world model.

    Attributes:
        patch_size: Latent patch size ``(t, h, w)`` of the transformer. Defaults to (1, 2, 2).
        num_attention_heads: Number of attention heads. Defaults to 24.
        attention_head_dim: Per-head dimension. Defaults to 128.
        in_channels: Video-latent channels consumed by the patch embedder. Defaults to 48.
        out_channels: Video-latent channels produced by the latent head. Defaults to 48.
        action_dim: Width of the multi-embodiment action space. Channels 0-6 are the
            left-arm end-effector pose, 7-13 the right arm, 14-27 the (unused) joint
            channels and 28/29 the grippers. Defaults to 30.
        text_dim: Width of the UMT5 hidden states. Defaults to 4096.
        freq_dim: Width of the sinusoidal timestep features. Defaults to 256.
        ffn_dim: Feed-forward inner dimension. Defaults to 14336.
        num_layers: Number of transformer blocks. Defaults to 30.
        cross_attn_norm: Layer-norm before text cross-attention. Defaults to True.
        eps: Layer-norm epsilon. Defaults to 1e-6.
        rope_max_seq_len: Maximum rotary sequence length. Defaults to 1024.
        attn_mode: Attention backend. ``"torch"`` (SDPA) is inference-only and always
            available; ``"flashattn"`` needs ``flash_attn``; ``"flex"`` is **required for
            training** because the flow-matching loss uses block-causal masks.
            Defaults to ``"torch"``.

        wan_pretrained_path: HuggingFace repo id or local directory holding the frozen
            ``vae/``, ``text_encoder/`` and ``tokenizer/`` sub-folders (~20 GB). These are
            pulled lazily on first use and are never written into the Studio checkpoint.
            Defaults to ``"robbyant/lingbot-va-base"``.
        dtype: Precision of the transformer, VAE and text encoder. Defaults to "bfloat16".
        text_encoder_device: Device for the frozen UMT5-XXL encoder. Keeping it on
            ``"cpu"`` frees ~11 GB of VRAM; it runs once per episode. Defaults to "cpu".

        obs_cam_keys: Camera keys in **fixed, order-sensitive** order — the first entry is
            the exterior/head view, the rest are wrist views. Both Studio names
            (``"image"``) and LeRobot names (``"observation.images.image"``) are accepted.
            Defaults to the LIBERO pair (agentview, eye-in-hand).
        image_hflip: Horizontally flip camera images before encoding, undoing the LIBERO
            env's extra flip so the orientation matches training. Defaults to False.
        camera_layout: How per-camera latents are assembled. ``"width_concat"``
            concatenates them along width (LIBERO); ``"robotwin_tshape"`` places a
            full-resolution head view below two half-resolution wrist views (RoboTwin).
            Defaults to "width_concat".

        n_obs_steps: Number of observation steps consumed per inference call. Defaults to 1.
        height: Camera height fed to the VAE. Defaults to 128.
        width: Camera width fed to the VAE. Defaults to 128.
        action_per_frame: Action sub-steps predicted per latent frame. Defaults to 4.
        frame_chunk_size: Latent frames predicted per autoregressive chunk. Defaults to 4.
        attn_window: Attention window in chunks for the streaming KV cache. Defaults to 30.
        num_inference_steps: Denoising steps for the video-latent stream. Defaults to 20.
        video_exec_step: Truncate the video denoising loop after this many steps
            (``-1`` runs the full loop). Defaults to -1.
        action_num_inference_steps: Denoising steps for the action stream. Defaults to 50.
        guidance_scale: Classifier-free guidance scale for the video stream. Defaults to 5.0.
        action_guidance_scale: Guidance scale for the action stream. Defaults to 1.0.
        snr_shift: Flow-matching SNR shift for the video stream. Defaults to 5.0.
        action_snr_shift: Flow-matching SNR shift for the action stream. Defaults to 0.05.
        max_sequence_length: Padded UMT5 prompt length. Defaults to 512.

        used_action_channel_ids: Which of the ``action_dim`` channels this checkpoint
            actually drives; also fixes the policy's output action dimension. LIBERO uses
            ``0-6`` (6-DoF end-effector delta + gripper). Defaults to ``(0, ..., 6)``.
        save_predicted_video: Keep the predicted (imagined) video latents on the model so
            they can be VAE-decoded into an MP4. Defaults to False.
        normalization_mode: Action (un)normalization method. The released checkpoints ship
            per-channel q01/q99, so ``"QUANTILES"`` is the default.

        optimizer_lr: Learning rate. Defaults to 1e-5.
        optimizer_betas: Adam beta coefficients. Defaults to (0.9, 0.95).
        optimizer_eps: Optimizer epsilon. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 1e-4.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 1.0.
        scheduler_warmup_steps: Linear warmup steps before the constant LR. Defaults to 1000.
    """

    # Wan transformer architecture
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    attn_mode: Literal["torch", "flashattn", "flex"] = "torch"

    # Frozen sub-models (VAE + UMT5 text encoder + tokenizer)
    wan_pretrained_path: str = "robbyant/lingbot-va-base"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    text_encoder_device: str = "cpu"

    # Observation cameras
    obs_cam_keys: tuple[str, ...] = ("observation.images.image", "observation.images.image2")
    image_hflip: bool = False
    camera_layout: Literal["width_concat", "robotwin_tshape"] = "width_concat"

    # Inference hyperparameters
    n_obs_steps: int = 1
    height: int = 128
    width: int = 128
    action_per_frame: int = 4
    frame_chunk_size: int = 4
    attn_window: int = 30
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    max_sequence_length: int = 512

    # Action space
    used_action_channel_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    save_predicted_video: bool = False
    normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES"

    # Optimizer / scheduler (AdamW + linear-warmup-then-constant, matching upstream)
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000

    def __post_init__(self) -> None:
        """Validate the configuration.

        Sequence fields are normalized to tuples so a config built from YAML or JSON
        (which yields lists) compares equal to one built from the defaults.

        Raises:
            ValueError: If the attention backend, camera layout, action channels or
                latent geometry are inconsistent.
        """
        for field_name in ("patch_size", "obs_cam_keys", "used_action_channel_ids", "optimizer_betas"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                object.__setattr__(self, field_name, tuple(value))

        if self.attn_mode not in {"torch", "flashattn", "flex"}:
            msg = f"attn_mode must be one of 'torch', 'flashattn', 'flex'; got {self.attn_mode!r}"
            raise ValueError(msg)

        if self.dtype not in {"bfloat16", "float16", "float32"}:
            msg = f"Invalid dtype: {self.dtype}"
            raise ValueError(msg)

        if not self.obs_cam_keys:
            msg = "obs_cam_keys must list at least one camera."
            raise ValueError(msg)

        tshape_cameras = 3
        if self.camera_layout == "robotwin_tshape" and len(self.obs_cam_keys) != tshape_cameras:
            msg = (
                f"camera_layout='robotwin_tshape' expects exactly {tshape_cameras} cameras "
                f"(head, left wrist, right wrist); got {len(self.obs_cam_keys)}."
            )
            raise ValueError(msg)

        if not self.used_action_channel_ids:
            msg = "used_action_channel_ids must select at least one action channel."
            raise ValueError(msg)

        if not all(0 <= i < self.action_dim for i in self.used_action_channel_ids):
            msg = f"used_action_channel_ids must be within [0, {self.action_dim}); got {self.used_action_channel_ids}"
            raise ValueError(msg)

        if self.action_per_frame % VAE_TEMPORAL_DOWNSAMPLE != 0 and self.action_per_frame > VAE_TEMPORAL_DOWNSAMPLE:
            msg = (
                f"action_per_frame ({self.action_per_frame}) must divide evenly into the VAE temporal "
                f"downsample ({VAE_TEMPORAL_DOWNSAMPLE}) or be a multiple of it."
            )
            raise ValueError(msg)

    @property
    def chunk_size(self) -> int:
        """Number of single-step actions produced per autoregressive chunk."""
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self) -> int:
        """Number of actions executed before the chunk is refilled (the whole chunk)."""
        return self.chunk_size

    @property
    def output_action_dim(self) -> int:
        """Dimension of the actions this policy emits (the used action channels)."""
        return len(self.used_action_channel_ids)

    @property
    def keyframe_stride(self) -> int:
        """Executed sub-steps between two observations kept as VAE keyframes."""
        return max(1, self.action_per_frame // VAE_TEMPORAL_DOWNSAMPLE)

    @property
    def latent_hw(self) -> tuple[int, int]:
        """Height and width of the assembled per-chunk video latent grid."""
        if self.camera_layout == "robotwin_tshape":
            # Full-resolution head below, two half-resolution wrists on top -> 1.5x height.
            return ((self.height // VAE_SPATIAL_DOWNSAMPLE) * 3) // 2, self.width // VAE_SPATIAL_DOWNSAMPLE
        height = self.height // VAE_SPATIAL_DOWNSAMPLE
        width = (self.width // VAE_SPATIAL_DOWNSAMPLE) * len(self.obs_cam_keys)
        return height, width

    @property
    def observation_delta_indices(self) -> list[int]:
        """Frame offsets of the observation clip each training sample must provide."""
        stride = self.keyframe_stride
        return list(range(0, self.frame_chunk_size * VAE_TEMPORAL_DOWNSAMPLE * stride, stride))

    @property
    def action_delta_indices(self) -> list[int]:
        """Action offsets each training sample must provide."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """LingBot-VA does not consume rewards."""
        return None
