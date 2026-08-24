# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the VLA-JEPA model.

VLA-JEPA combines a Qwen3-VL vision-language backbone, a flow-matching DiT action head and a
self-supervised V-JEPA2 video world model used as an auxiliary training loss.

For CLI usage, use the YAML config in `configs/physicalai/vla_jepa.yaml`:

    physicalai fit --config configs/physicalai/vla_jepa.yaml

Example (API):
    >>> from physicalai.policies.vla_jepa import VLAJEPAConfig
    >>> config = VLAJEPAConfig(chunk_size=16, n_action_steps=8, enable_world_model=False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from physicalai.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLAJEPAConfig(Config):
    """Configuration for the VLA-JEPA vision-language-action model.

    Attributes:
        n_obs_steps: Number of observation steps to use. Defaults to 1.
        chunk_size: Size of the predicted action chunk. Defaults to 7.
        n_action_steps: Number of action steps executed per model invocation. Defaults to 7.
        qwen_model_name: Name or path of the Qwen3-VL backbone.
            Defaults to "Qwen/Qwen3-VL-2B-Instruct".
        jepa_encoder_name: Name or path of the V-JEPA2 video encoder used by the world model.
            Defaults to "facebook/vjepa2-vitl-fpc64-256".
        freeze_qwen: Whether to freeze the Qwen backbone. Freezing it also disables the world
            model, since no gradient would reach it. Defaults to False.
        enable_world_model: Whether to build and train the V-JEPA2 world model. Inference never
            uses it. Defaults to True.
        reinit_modules: Key prefixes allowed to have shape mismatches when loading pretrained
            weights, re-initialised from scratch instead. Enables cross-embodiment transfer, e.g.
            ["model.action_model.action_encoder", "model.action_model.state_encoder"].
            Defaults to None.
        tokenizer_padding_side: Padding side for the Qwen tokenizer. Defaults to "left".
        prompt_template: Chat-template prompt the checkpoints were trained with.
        special_action_token: Format string for the per-timestep action tokens.
            Defaults to "<|action_{}|>".
        embodied_action_token: Token whose hidden states condition the action head.
            Defaults to "<|embodied_action|>".
        action_dim: Action dimensionality. Resolved from dataset stats during setup. Defaults to 7.
        state_dim: State dimensionality. Resolved from dataset stats during setup. Defaults to 8.
        state_normalization: Normalization applied to the robot state. Defaults to "MEAN_STD".
        action_normalization: Normalization applied to actions. Defaults to "MIN_MAX".
        use_relative_actions: Whether to convert absolute actions to relative (action -= state)
            during preprocessing and reverse it during postprocessing. Defaults to False.
        relative_exclude_joints: Joint names kept absolute when `use_relative_actions` is set.
            An empty list makes every dimension relative. Defaults to ["gripper"].
        action_feature_names: Per-dimension action names, populated from dataset metadata during
            setup and used to resolve the gripper index. Defaults to None.
        num_action_tokens_per_timestep: Action tokens the world model consumes per timestep.
            Defaults to 8.
        num_embodied_action_tokens_per_instruction: Number of embodied-action conditioning tokens.
            Defaults to 32.
        num_inference_timesteps: Flow-matching integration steps at inference. Defaults to 4.
        action_hidden_size: Hidden width of the action head's encoder/decoder MLPs. Defaults to 1024.
        action_model_type: DiT preset controlling the attention geometry ("DiT-B", "DiT-L",
            "DiT-test"). Defaults to "DiT-B".
        action_num_layers: Number of DiT transformer blocks. Defaults to 16.
        action_num_heads: Attention heads, overriding the preset when set. Defaults to None.
        action_attention_head_dim: Attention head dimension, overriding the preset when set.
            Defaults to None.
        action_dropout: Dropout used inside the DiT. Defaults to 0.2.
        action_num_timestep_buckets: Number of discrete flow-matching timestep buckets.
            Defaults to 1000.
        action_noise_beta_alpha: Alpha of the Beta distribution used for timestep sampling.
            Defaults to 1.5.
        action_noise_beta_beta: Beta of the Beta distribution used for timestep sampling.
            Defaults to 1.0.
        action_noise_s: Scaling applied to the sampled timesteps. Defaults to 0.999.
        action_max_seq_len: Size of the action head's learned position-embedding table. Kept at
            1024 to match the published checkpoints. Defaults to 1024.
        num_video_frames: Total video frames loaded per sample for the world model. Defaults to 8.
        predictor_depth: Number of world-model predictor blocks. Defaults to 12.
        predictor_num_heads: Attention heads in the world-model predictor. Defaults to 8.
        predictor_mlp_ratio: MLP expansion ratio in the world-model predictor. Defaults to 4.0.
        predictor_dropout: Dropout in the world-model predictor. Defaults to 0.0.
        world_model_loss_weight: Weight of the world-model loss in the total loss. Defaults to 0.1.
        jepa_tubelet_size: Temporal tubelet size of the JEPA encoder. When the world model is
            enabled the encoder's own `config.tubelet_size` is authoritative and this is only used
            for the `num_video_frames` sanity check. Defaults to 2.
        world_model_num_views: Camera views the world-model predictor is built for (extra views
            trimmed, missing ones padded with the first). Baked into checkpoint shapes. None falls
            back to `jepa_tubelet_size`, which is what the published checkpoints encode.
            Defaults to None.
        repeated_diffusion_steps: Independent noise draws per batch item (CogACT-style).
            Defaults to 8.
        causal_world_model_context: Whether to encode the world-model context causally instead of
            slicing it from the shared bidirectional pass. Defaults to False.
        resize_images_to: Target (height, width) images are resized to before the backbone.
            Defaults to None.
        binarize_gripper_action: Whether to binarize the gripper dimension after unnormalization.
            Only correct for LIBERO's action convention. Defaults to False.
        pre_snap_gripper_action: Whether to snap the gripper dimension to {0, 1} before
            unnormalization. Only correct for LIBERO's action convention. Defaults to False.
        clip_normalized_actions: Whether to clip normalized actions to [-1, 1] before
            unnormalization. Ignored unless actions use MIN_MAX normalization. Defaults to True.
        gripper_dim: Index of the gripper in the action vector. Prefer setting
            `gripper_joint_names`, which resolves the index from dataset metadata. Defaults to 6.
        gripper_threshold: Threshold used by the gripper post-processing steps. Defaults to 0.5.
        gripper_joint_names: Action-dimension names identifying the gripper. When these match
            `action_feature_names`, the resolved index wins over `gripper_dim`.
            Defaults to ["gripper"].
        torch_dtype: Compute dtype for the pretrained backbones. Defaults to "bfloat16".
        optimizer_lr: Learning rate for the optimizer. Defaults to 1e-4.
        optimizer_betas: Beta coefficients for AdamW. Defaults to (0.9, 0.95).
        optimizer_eps: Epsilon for numerical stability. Defaults to 1e-8.
        optimizer_weight_decay: Weight decay coefficient. Defaults to 1e-10.
        optimizer_grad_clip_norm: Maximum gradient norm. Defaults to 10.0.
        scheduler_warmup_steps: Warmup steps for the scheduler. Defaults to 1000.
        scheduler_decay_steps: Decay steps for the scheduler. Defaults to 30000.
        scheduler_decay_lr: Final learning rate after decay. Defaults to 2.5e-6.
    """

    n_obs_steps: int = 1
    chunk_size: int = 7
    n_action_steps: int = 7

    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"
    freeze_qwen: bool = False
    enable_world_model: bool = True
    reinit_modules: list[str] | None = None

    tokenizer_padding_side: str = "left"
    prompt_template: str = (
        "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and "
        "produce the corresponding policy actions {e_actions}."
    )
    special_action_token: str = "<|action_{}|>"  # noqa: S105
    embodied_action_token: str = "<|embodied_action|>"  # noqa: S105

    action_dim: int = 7
    state_dim: int = 8

    state_normalization: str = "MEAN_STD"
    action_normalization: str = "MIN_MAX"

    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None

    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens_per_instruction: int = 32
    num_inference_timesteps: int = 4

    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 16
    action_num_heads: int | None = None
    action_attention_head_dim: int | None = None
    action_dropout: float = 0.2
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999
    action_max_seq_len: int = 1024

    num_video_frames: int = 8
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    world_model_loss_weight: float = 0.1
    jepa_tubelet_size: int = 2
    world_model_num_views: int | None = None
    repeated_diffusion_steps: int = 8
    causal_world_model_context: bool = False

    resize_images_to: tuple[int, int] | None = None
    binarize_gripper_action: bool = False
    pre_snap_gripper_action: bool = False
    clip_normalized_actions: bool = True
    gripper_dim: int = 6
    gripper_threshold: float = 0.5
    gripper_joint_names: list[str] = field(default_factory=lambda: ["gripper"])
    torch_dtype: str = "bfloat16"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """Validate the configuration and resolve mutually exclusive flags.

        Raises:
            ValueError: If `n_action_steps` exceeds `chunk_size`, or if `num_video_frames` is too
                small for the world model to have both a context and a ground-truth position.
        """
        if self.freeze_qwen and self.enable_world_model:
            # Freezing the qwen backbone makes world model training irrelevant: no gradient flows.
            object.__setattr__(self, "enable_world_model", False)
        if self.freeze_qwen:
            logger.warning(
                "freeze_qwen=True: action-head conditioning is read from %s positions at the last "
                "decoder layer. These learned readouts stay fixed from the source checkpoint and "
                "cannot adapt to a new embodiment while the Qwen backbone is frozen, so conditioning "
                "quality may degrade under domain shift.",
                self.embodied_action_token,
            )
        if self.n_action_steps > self.chunk_size:
            msg = (
                f"The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                f"for `chunk_size`."
            )
            raise ValueError(msg)
        if self.num_video_frames < 2 * self.jepa_tubelet_size:
            msg = (
                f"`num_video_frames` ({self.num_video_frames}) must be >= 2 * `jepa_tubelet_size` "
                f"({self.jepa_tubelet_size}) to have at least one context and one ground-truth "
                f"temporal position."
            )
            raise ValueError(msg)

    @property
    def num_world_model_views(self) -> int:
        """Camera views the world model predictor is built for.

        Returns:
            `world_model_num_views` when set, otherwise `jepa_tubelet_size`, which is what the
            published checkpoints encode.
        """
        return self.world_model_num_views or self.jepa_tubelet_size

    @property
    def resolved_gripper_dim(self) -> int:
        """Gripper index, resolved from `action_feature_names` when possible.

        Returns:
            The index of the first action dimension whose name matches `gripper_joint_names`,
            falling back to the raw `gripper_dim` when dataset metadata is unavailable.
        """
        if not self.action_feature_names or not self.gripper_joint_names:
            return self.gripper_dim
        wanted = [name.lower() for name in self.gripper_joint_names if name]
        for index, name in enumerate(self.action_feature_names):
            lowered = str(name).lower()
            if any(token == lowered or token in lowered for token in wanted):
                return index
        return self.gripper_dim

    @property
    def observation_delta_indices(self) -> list[int]:
        """Observation indices relative to the current timestep.

        Returns:
            `[0]` when the world model is disabled, since only it consumes frames past index 0.
            Otherwise the frame window, strided across the chunk when the chunk is longer than
            `num_video_frames` so the world model sees dynamics over the whole horizon.
        """
        if not self.enable_world_model:
            return [0]
        if self.num_video_frames >= self.chunk_size:
            return list(range(self.num_video_frames))
        stride = (self.chunk_size - 1) // (self.num_video_frames - 1)
        return [i * stride for i in range(self.num_video_frames)]

    @property
    def action_delta_indices(self) -> list[int]:
        """Action indices relative to the current timestep.

        Returns:
            One index per step of the predicted chunk.
        """
        return list(range(self.chunk_size))
