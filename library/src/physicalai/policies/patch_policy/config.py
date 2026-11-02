# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Serializable configuration for Patch Policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

from physicalai.config import Config

from physicalai.data import Feature, FeatureType

type FeatureEncoderName = Literal["dinov2", "webssl", "resnet18"]
type EncoderOutput = Literal["patches", "cls", "average"]
type ActionHeadName = Literal["diffusion", "behavior_transformer"]


def resolve_action_dim(output_features: list[Feature]) -> int:
    """Return the flattened width of the configured action features.

    Args:
        output_features: Action features whose dimensions are combined.

    Returns:
        The total flattened action width.

    Raises:
        ValueError: If no features are provided or a feature is not a valid action.
    """
    if not output_features:
        msg = "Patch Policy requires at least one output feature."
        raise ValueError(msg)
    width = 0
    for feature in output_features:
        if feature.ftype is not FeatureType.ACTION or not feature.shape:
            msg = f"Output feature {feature.name!r} must be an action with a non-empty shape."
            raise ValueError(msg)
        width += prod(feature.shape)
    return width


@dataclass(frozen=True)
class PatchPolicyConfig(Config):
    """Feature, architecture, and horizon contract for Patch Policy.

    Args:
        input_features: Observation features available to the policy.
        output_features: Action features predicted by the policy.
        action_dim: Flattened width of all output action features.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        n_action_steps: Number of predicted actions executed per invocation.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_output: Visual representation extracted from the encoder.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        image_size: Encoder input height and width, defaulting to (224, 224).
        action_head: Action-head architecture used to predict action chunks.
        hidden_dim: Hidden width of the action head.
        num_layers: Number of transformer layers in the action head.
        num_heads: Number of attention heads in each transformer layer.
        dropout: Dropout probability used by transformer layers.
        embedding_dropout: Dropout applied to diffusion action and context embeddings.
        num_train_timesteps: Number of diffusion timesteps used during training.
        num_inference_steps: Number of denoising steps used during inference.
        beta_schedule: Noise schedule used by the diffusion head.
        prediction_type: Diffusion target represented by the model output.
        clip_sample: Whether to clip diffusion samples after each scheduler step.
        clip_sample_range: Symmetric range used when clipping diffusion samples.
        eta: DDIM stochasticity coefficient used during inference.
        use_random_input_noise: Whether diffusion sampling starts from random noise.
        vq_latent_dim: Latent width of the Behavior Transformer quantizer.
        vq_num_embeddings: Number of codebook entries in each residual VQ group.
        vq_num_groups: Number of residual vector-quantization groups.
        vq_action_scale: Action scale used inside the Behavior Transformer VQ-VAE.
        offset_loss_weight: Weight applied to continuous action-offset loss.
        secondary_code_weight: Relative loss weight for residual code groups after the first.
        focal_loss_gamma: Focusing exponent used by the code-prediction loss.
    """

    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int

    n_obs_steps: int = 5
    chunk_size: int = 5
    n_action_steps: int = 1

    encoder_name: FeatureEncoderName = "dinov2"
    encoder_output: EncoderOutput = "patches"
    encoder_pretrained: bool = True
    image_size: tuple[int, int] = (224, 224)
    action_head: ActionHeadName = "behavior_transformer"

    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    embedding_dropout: float = 0.0

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2"
    prediction_type: Literal["epsilon", "sample"] = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    eta: float = 1.0
    use_random_input_noise: bool = True

    vq_latent_dim: int = 512
    vq_num_embeddings: int = 16
    vq_num_groups: int = 2
    vq_action_scale: float = 500.0
    vqvae_iters: int = 1000
    vqvae_batch_size: int = 1024
    offset_loss_weight: float = 10.0
    secondary_code_weight: float = 0.5
    focal_loss_gamma: float = 2.0

    def __post_init__(self) -> None:
        """Validate values that affect model shapes.

        Raises:
            ValueError: If features, dimensions, horizons, or VQ settings are invalid.
        """
        if not any(feature.ftype is FeatureType.VISUAL for feature in self.input_features):
            msg = "Patch Policy requires at least one visual input feature."
            raise ValueError(msg)
        if self.action_dim != resolve_action_dim(self.output_features):
            msg = "action_dim does not match the configured output features."
            raise ValueError(msg)
        if self.n_obs_steps <= 0 or self.chunk_size <= 0:
            msg = "n_obs_steps and chunk_size must be positive."
            raise ValueError(msg)
        if not 0 < self.n_action_steps <= self.chunk_size:
            msg = "n_action_steps must be in [1, chunk_size]."
            raise ValueError(msg)
        if self.hidden_dim % self.num_heads:
            msg = "hidden_dim must be divisible by num_heads."
            raise ValueError(msg)
        if self.num_layers <= 0 or self.num_heads <= 0:
            msg = "num_layers and num_heads must be positive."
            raise ValueError(msg)
        if self.vq_num_groups <= 0 or self.vq_num_embeddings <= 0:
            msg = "VQ group and embedding counts must be positive."
            raise ValueError(msg)
        if self.vq_action_scale <= 0:
            msg = "vq_action_scale must be positive."
            raise ValueError(msg)
        if self.vqvae_iters <= 0 or self.vqvae_batch_size <= 0:
            msg = "vqvae_iters and vqvae_batch_size must be positive."
            raise ValueError(msg)
