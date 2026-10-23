# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Encoder-to-head Patch Policy model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from physicalai.policies.base import Model
from physicalai.policies.utils import in_episode_bound, reduce_losses

from .encoder import TimmFeatureEncoder
from .heads import PatchBehaviorTransformerHead, PatchDiffusionHead

if TYPE_CHECKING:
    from .config import ActionHeadName, EncoderOutput, FeatureEncoderName, PatchPolicyConfig


class PatchPolicyModel(Model):
    """A visual feature encoder followed by an interchangeable action head.

    Args:
        action_dim: Width of each action timestep.
        num_views: Number of configured camera views.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        image_size: Input image height and width.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_output: Visual representation extracted from the encoder.
        encoder_pretrained: Whether to initialize the encoder with pretrained weights.
        freeze_encoder: Whether to prevent updates to encoder parameters.
        action_head: Action-head architecture used to predict action chunks.
        hidden_dim: Hidden width of the action head.
        num_layers: Number of transformer layers in the action head.
        num_heads: Number of attention heads in each transformer layer.
        dropout: Dropout probability used by transformer layers.
        num_train_timesteps: Number of diffusion timesteps used during training.
        num_inference_steps: Number of denoising steps used during inference.
        beta_schedule: Noise schedule used by the diffusion head.
        prediction_type: Diffusion target represented by the model output.
        clip_sample: Whether to clip diffusion samples after scheduler steps.
        clip_sample_range: Symmetric range used when clipping diffusion samples.
        eta: DDIM stochasticity coefficient used during inference.
        use_random_input_noise: Whether diffusion sampling starts from random noise.
        vq_latent_dim: Latent width of the Behavior Transformer quantizer.
        vq_num_embeddings: Number of codebook entries in each residual VQ group.
        vq_num_groups: Number of residual vector-quantization groups.
        vq_reconstruction_weight: Weight applied to the VQ reconstruction loss.
        vq_commitment_weight: Weight applied to the VQ commitment loss.
        offset_loss_weight: Weight applied to continuous action-offset loss.
        secondary_code_weight: Relative loss weight for residual code groups after the first.
        focal_loss_gamma: Focusing exponent used by the code-prediction loss.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        action_dim: int,
        num_views: int,
        n_obs_steps: int = 1,
        chunk_size: int = 16,
        image_size: tuple[int, int] = (224, 224),
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_pretrained: bool = True,
        freeze_encoder: bool = True,
        action_head: ActionHeadName = "diffusion",
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
        vq_latent_dim: int = 128,
        vq_num_embeddings: int = 16,
        vq_num_groups: int = 2,
        vq_reconstruction_weight: float = 1.0,
        vq_commitment_weight: float = 0.25,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self.encoder = TimmFeatureEncoder(
            encoder_name,
            encoder_output,
            pretrained=encoder_pretrained,
            frozen=freeze_encoder,
            image_size=image_size,
        )
        if action_head == "diffusion":
            self.action_head = PatchDiffusionHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_inference_steps,
                beta_schedule=beta_schedule,  # type: ignore[arg-type]
                prediction_type=prediction_type,  # type: ignore[arg-type]
                clip_sample=clip_sample,
                clip_sample_range=clip_sample_range,
                eta=eta,
                use_random_input_noise=use_random_input_noise,
            )
        else:
            self.action_head = PatchBehaviorTransformerHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                n_obs_steps=n_obs_steps,
                n_patches=self.encoder.num_tokens * num_views,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                latent_dim=vq_latent_dim,
                num_embeddings=vq_num_embeddings,
                groups=vq_num_groups,
                reconstruction_weight=vq_reconstruction_weight,
                commitment_weight=vq_commitment_weight,
                offset_loss_weight=offset_loss_weight,
                secondary_code_weight=secondary_code_weight,
                focal_loss_gamma=focal_loss_gamma,
            )

    @classmethod
    def from_config(cls, config: PatchPolicyConfig) -> PatchPolicyModel:
        """Build the complete model architecture from one serialized config.

        Args:
            config: Resolved Patch Policy configuration.

        Returns:
            A model containing the encoder and configured action head.
        """
        num_views = sum(
            feature.ftype is not None and feature.ftype.value == "VISUAL" for feature in config.input_features
        )
        return cls(
            action_dim=config.action_dim,
            num_views=num_views,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            image_size=config.image_size,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_pretrained=config.encoder_pretrained,
            freeze_encoder=config.freeze_encoder,
            action_head=config.action_head,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            eta=config.eta,
            use_random_input_noise=config.use_random_input_noise,
            vq_latent_dim=config.vq_latent_dim,
            vq_num_embeddings=config.vq_num_embeddings,
            vq_num_groups=config.vq_num_groups,
            vq_reconstruction_weight=config.vq_reconstruction_weight,
            vq_commitment_weight=config.vq_commitment_weight,
            offset_loss_weight=config.offset_loss_weight,
            secondary_code_weight=config.secondary_code_weight,
            focal_loss_gamma=config.focal_loss_gamma,
        )

    def _context(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode batch images into flattened visual context tokens.

        Args:
            batch: Model batch containing preprocessed images.

        Returns:
            Action-head context containing visual tokens.
        """
        images = batch["images"]
        tokens = self.encoder(images)
        return {"tokens": tokens.flatten(1, 3)}

    def forward(self, batch: dict[str, Any]) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Compute training loss or sample actions according to module mode.

        Args:
            batch: Preprocessed model batch.

        Returns:
            Training loss and metrics in training mode, otherwise an action chunk.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.predict_action_chunk(batch)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the reduced training loss and metrics.

        Args:
            batch: Preprocessed batch containing images and target actions.

        Returns:
            The scalar loss and detached logging metrics.
        """
        losses = self.action_head.compute_loss(batch["action"], self._context(batch))
        loss = reduce_losses(losses, in_episode_bound(batch))
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute action mean-squared error for validation.

        Args:
            batch: Preprocessed batch containing images and target actions.

        Returns:
            The scalar validation loss and detached logging metrics.
        """
        actions = self.action_head.sample(self._context(batch))
        loss = reduce_losses(F.mse_loss(actions, batch["action"], reduction="none"), in_episode_bound(batch))
        return loss, {"loss": loss.detach(), "action_mse": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Predict one normalized action chunk.

        Args:
            batch: Preprocessed batch containing images.

        Returns:
            A normalized action chunk for each batch item.
        """
        return self.action_head.sample(self._context(batch))

    @property
    def action_delta_indices(self) -> list[int]:
        """Dataset offsets required for action chunks.

        Returns:
            Consecutive offsets covering the action horizon.
        """
        return list(range(self._chunk_size))

    @property
    def observation_delta_indices(self) -> list[int]:
        """Dataset offsets required for observation context.

        Returns:
            Historical offsets ending at the current timestep.
        """
        return list(range(1 - self._n_obs_steps, 1))

    @property
    def reward_delta_indices(self) -> None:
        """Reward dataset offsets, which Patch Policy does not require.

        Returns:
            None.
        """
        return None
