# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR2004

"""Resolved architecture and feature contract for VLA-JEPA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from physicalai.config import Config

from physicalai.data import Feature, FeatureType
from physicalai.policies.mixins.peft import PeftConfigMixin

if TYPE_CHECKING:
    from collections.abc import Mapping


def feature_dimension(features: list[Feature], kind: FeatureType) -> int:
    """Return a single vector feature's width, or zero when it is absent.

    Raises:
        ValueError: If the feature is not a single vector.
    """
    selected = [feature for feature in features if feature.ftype == kind]
    if not selected:
        return 0
    if len(selected) != 1 or not selected[0].shape or len(selected[0].shape) != 1 or selected[0].shape[0] <= 0:
        msg = f"VLA-JEPA requires a single one-dimensional {kind} feature."
        raise ValueError(msg)
    return selected[0].shape[0]


@dataclass(frozen=True)
class VLAJEPAConfig(PeftConfigMixin, Config):
    """Complete model/processor configuration, independent of pretrained downloads."""

    format_version: int = 2
    input_features: list[Feature] = field(default_factory=list)
    output_features: list[Feature] = field(default_factory=list)
    qwen_config: dict[str, Any] = field(default_factory=dict)
    dtype: str = "bfloat16"
    tokenizer_json: str = ""
    image_processor_config: dict[str, Any] = field(default_factory=dict)
    chat_template: str = ""
    embodied_action_token_id: int = 0
    prompt_template: str = (
        "Your task is {instruction}. Infer the temporal dynamics from frames {actions} "
        "and produce the corresponding policy actions {e_actions}."
    )
    num_latent_tokens: int = 3
    num_special_action_tokens: int = 28
    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens: int = 32
    chunk_size: int = 7
    n_action_steps: int = 7
    action_dim: int = 7
    state_dim: int = 8
    action_hidden_size: int = 1024
    action_num_layers: int = 16
    action_num_heads: int = 12
    action_attention_head_dim: int = 64
    action_dropout: float = 0.2
    num_target_vision_tokens: int = 32
    action_max_seq_len: int = 1024
    num_inference_steps: int = 4
    num_timestep_buckets: int = 1000
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    repeated_diffusion_steps: int = 8
    image_resolution: tuple[int, int] = (224, 224)
    enable_world_model: bool = True
    world_model_config: dict[str, Any] = field(default_factory=dict)
    video_processor_config: dict[str, Any] = field(default_factory=dict)
    num_video_frames: int = 8
    world_model_num_views: int = 2
    world_model_loss_weight: float = 0.1
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    state_normalization: str = "MEAN_STD"
    action_normalization: str = "MIN_MAX"
    clip_normalized_actions: bool = True
    pre_snap_gripper: bool = False
    binarize_gripper: bool = False
    gripper_dim: int = 6
    gripper_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, strict: bool = True) -> Self:
        """Return a config, preserving action-only behavior of version-one checkpoints."""
        if data.get("format_version") == 1:
            data = {**data, "format_version": 2, "enable_world_model": False}
        return super().from_dict(data, strict=strict)

    def __post_init__(self) -> None:
        """Reject invalid horizons, feature contracts, and architecture settings.

        Raises:
            ValueError: If the configuration is inconsistent.
        """
        super().__post_init__()
        if self.format_version != 2:
            msg = f"Unsupported VLA-JEPA config version: {self.format_version}"
            raise ValueError(msg)
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            msg = f"Unsupported model dtype: {self.dtype}"
            raise ValueError(msg)
        positive = (
            self.chunk_size,
            self.n_action_steps,
            self.action_dim,
            self.action_hidden_size,
            self.action_num_layers,
            self.action_num_heads,
            self.action_attention_head_dim,
            self.num_inference_steps,
            self.num_timestep_buckets,
            self.repeated_diffusion_steps,
            self.num_embodied_action_tokens,
            self.num_action_tokens_per_timestep,
            self.num_target_vision_tokens,
            self.action_max_seq_len,
            *self.image_resolution,
        )
        if any(value <= 0 for value in positive) or self.state_dim < 0:
            msg = "VLA-JEPA dimensions, horizons, and step counts must be positive."
            raise ValueError(msg)
        if self.n_action_steps > self.chunk_size or self.chunk_size > self.action_max_seq_len:
            msg = "Expected n_action_steps <= chunk_size <= action_max_seq_len."
            raise ValueError(msg)
        if self.num_latent_tokens < 1 or self.num_latent_tokens > self.num_special_action_tokens:
            msg = "The latent prompt must fit the checkpoint's action-token vocabulary."
            raise ValueError(msg)
        if not 0 <= self.action_dropout < 1 or min(self.noise_beta_alpha, self.noise_beta_beta, self.noise_s) <= 0:
            msg = "Invalid dropout or flow-matching noise distribution."
            raise ValueError(msg)
        if self.action_num_heads * self.action_attention_head_dim % 2:
            msg = "The action embedding width must be even."
            raise ValueError(msg)
        self._validate_world_model()
        self._validate_features()

    def _validate_world_model(self) -> None:
        if not math.isfinite(self.world_model_loss_weight) or self.world_model_loss_weight < 0:
            msg = "world_model_loss_weight must be finite and nonnegative."
            raise ValueError(msg)
        if (
            min(
                self.num_video_frames,
                self.world_model_num_views,
                self.predictor_depth,
                self.predictor_num_heads,
                self.predictor_mlp_ratio,
            )
            <= 0
        ):
            msg = "World-model dimensions and video length must be positive."
            raise ValueError(msg)
        if self.enable_world_model and self.world_model_config:
            patch = self.world_model_config["patch_size"]
            crop = self.world_model_config["crop_size"]
            if type(patch) is not int or type(crop) is not int or min(patch, crop) <= 0 or crop % patch:
                msg = "The world model requires a square crop divisible by its scalar patch size."
                raise ValueError(msg)
            tubelet = self.world_model_config["tubelet_size"]
            width = self.world_model_config["hidden_size"]
            if tubelet <= 0 or self.num_video_frames != (self.num_latent_tokens + 1) * tubelet:
                msg = "Video length must match the pretrained latent-action prompt and tubelet size."
                raise ValueError(msg)
            if width % self.predictor_num_heads or width // self.predictor_num_heads < 6:
                msg = "World-predictor attention requires divisible heads of width at least six."
                raise ValueError(msg)

    def _validate_features(self) -> None:
        for features in (self.input_features, self.output_features):
            names = [feature.name for feature in features]
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                msg = "Feature names must be nonempty and unique."
                raise ValueError(msg)
        if self.input_features:
            images = [feature for feature in self.input_features if feature.ftype == FeatureType.VISUAL]
            if not images or any(
                not feature.shape or len(feature.shape) != 3 or feature.shape[0] != 3 for feature in images
            ):
                msg = "VLA-JEPA requires at least one CHW RGB image feature."
                raise ValueError(msg)
            width = feature_dimension(self.input_features, FeatureType.STATE)
            if width and width != self.state_dim:
                msg = "State feature width does not match state_dim."
                raise ValueError(msg)
        if self.output_features and feature_dimension(self.output_features, FeatureType.ACTION) != self.action_dim:
            msg = "Action feature width does not match action_dim."
            raise ValueError(msg)
        for mode in (self.state_normalization, self.action_normalization):
            if mode not in {"IDENTITY", "MIN_MAX", "MEAN_STD", "QUANTILES"}:
                msg = f"Unsupported normalization mode: {mode}"
                raise ValueError(msg)
        if (self.pre_snap_gripper or self.binarize_gripper) and not 0 <= self.gripper_dim < self.action_dim:
            msg = "The gripper dimension is outside the action vector."
            raise ValueError(msg)
