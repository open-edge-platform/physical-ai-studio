# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Encoder-to-head Patch Policy model."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from physicalai.data.observation import IMAGES
from physicalai.policies.base import Model
from physicalai.policies.utils import in_episode_bound, reduce_losses

from .components.encoder import TimmFeatureEncoder
from .components.heads import BehaviourTransformerHead, DiffusionHead
from .processors import resolve_camera_name

if TYPE_CHECKING:
    from .config import ActionHeadName, EncoderOutput, FeatureEncoderName, PatchPolicyConfig


class PatchPolicyModel(Model):
    """A visual feature encoder followed by an interchangeable action head.

    Args:
        action_dim: Width of each action timestep.
        camera_names: Camera names in their configured view order.
        n_obs_steps: Number of observation timesteps used as context.
        chunk_size: Number of action timesteps predicted per chunk.
        encoder_name: Name of the timm visual encoder architecture.
        encoder_output: Visual representation extracted from the encoder.
        encoder_config: Resolved WebSSL architecture configuration; required for WebSSL.
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
        clip_sample: Whether to clip diffusion samples after scheduler steps.
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

    def __init__(  # noqa: PLR0913
        self,
        *,
        action_dim: int,
        camera_names: list[str],
        n_obs_steps: int = 5,
        chunk_size: int = 5,
        encoder_name: FeatureEncoderName = "dinov2",
        encoder_output: EncoderOutput = "patches",
        encoder_config: dict[str, Any] | None = None,
        image_size: tuple[int, int] = (224, 224),
        action_head: ActionHeadName = "behavior_transformer",
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
        vq_latent_dim: int = 512,
        vq_num_embeddings: int = 16,
        vq_num_groups: int = 2,
        vq_action_scale: float = 500.0,
        offset_loss_weight: float = 10.0,
        secondary_code_weight: float = 0.5,
        focal_loss_gamma: float = 2.0,
    ) -> None:
        """Initialize the encoder and action head.

        Raises:
            ValueError: If camera names are empty or duplicated.
        """
        super().__init__()
        if not camera_names or len(camera_names) != len(set(camera_names)):
            message = f"camera_names must be non-empty and unique, got {camera_names!r}."
            raise ValueError(message)
        self.camera_names = tuple(camera_names)
        self._n_obs_steps = n_obs_steps
        self._chunk_size = chunk_size
        self.encoder = TimmFeatureEncoder(
            encoder_name,
            encoder_output,
            encoder_config=encoder_config,
            image_size=image_size,
        )
        if action_head == "diffusion":
            self.action_head = DiffusionHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                n_obs_steps=n_obs_steps,
                n_patches=self.encoder.num_tokens * len(camera_names),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                embedding_dropout=embedding_dropout,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_inference_steps,
                beta_schedule=beta_schedule,  # type: ignore[arg-type]
                prediction_type=prediction_type,  # type: ignore[arg-type]
                clip_sample=clip_sample,
                clip_sample_range=clip_sample_range,
                eta=eta,
                use_random_input_noise=use_random_input_noise,
            )
            self.ema_action_head: DiffusionHead | None = copy.deepcopy(self.action_head)
            self.ema_action_head.requires_grad_(requires_grad=False).eval()
            self.register_buffer("ema_optimization_step", torch.tensor(0, dtype=torch.int64))
        else:
            self.ema_action_head = None
            self.action_head = BehaviourTransformerHead(
                self.encoder.output_dim,
                action_dim,
                chunk_size,
                n_obs_steps=n_obs_steps,
                n_patches=self.encoder.num_tokens * len(camera_names),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                latent_dim=vq_latent_dim,
                num_embeddings=vq_num_embeddings,
                groups=vq_num_groups,
                action_scale=vq_action_scale,
                offset_loss_weight=offset_loss_weight,
                secondary_code_weight=secondary_code_weight,
                focal_loss_gamma=focal_loss_gamma,
            )

    @classmethod
    def from_config(cls, config: PatchPolicyConfig) -> PatchPolicyModel:
        """Build the complete model architecture offline from one serialized config.

        Args:
            config: Resolved Patch Policy configuration.

        Returns:
            A model containing the encoder and configured action head.
        """
        camera_names = [
            resolve_camera_name(feature)
            for feature in config.input_features
            if feature.ftype is not None and feature.ftype.value == "VISUAL"
        ]
        return cls(
            action_dim=config.action_dim,
            camera_names=camera_names,
            n_obs_steps=config.n_obs_steps,
            chunk_size=config.chunk_size,
            encoder_name=config.encoder_name,
            encoder_output=config.encoder_output,
            encoder_config=config.encoder_config,
            image_size=config.image_size,
            action_head=config.action_head,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            embedding_dropout=config.embedding_dropout,
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
            vq_action_scale=config.vq_action_scale,
            offset_loss_weight=config.offset_loss_weight,
            secondary_code_weight=config.secondary_code_weight,
            focal_loss_gamma=config.focal_loss_gamma,
        )

    def _encode_observations(self, batch: dict[str, Any]) -> Tensor:
        """Encode batch images into flattened visual context tokens.

        Args:
            batch: Model batch containing preprocessed images.

        Returns:
            Visual tokens shaped [B, T, V * P, E].

        Raises:
            KeyError: If a configured camera tensor is missing.
            TypeError: If images are not provided as a mapping.
            ValueError: If camera shapes, names, or temporal lengths are invalid.
        """
        images = batch.get(IMAGES)
        if images is None:
            images = {
                key.removeprefix(f"{IMAGES}."): value for key, value in batch.items() if key.startswith(f"{IMAGES}.")
            }
        if not isinstance(images, Mapping):
            message = "Patch Policy expects images as a dictionary of camera tensors."
            raise TypeError(message)

        views: list[Tensor] = []
        reference_shape: torch.Size | None = None
        for camera_name in self.camera_names:
            image = images.get(camera_name)
            if not isinstance(image, Tensor):
                message = f"Missing tensor for configured camera {camera_name!r}."
                raise KeyError(message)
            if image.ndim != 5:  # noqa: PLR2004
                message = f"Camera {camera_name!r} must have shape [B, T, C, H, W], got {tuple(image.shape)}."
                raise ValueError(message)
            if image.shape[1] != self._n_obs_steps:
                message = f"Expected {self._n_obs_steps} observation steps, got {image.shape[1]} for {camera_name!r}."
                raise ValueError(message)
            if reference_shape is None:
                reference_shape = image.shape
            elif image.shape != reference_shape:
                message = (
                    "All Patch Policy camera tensors must have matching [B, T, C, H, W] shapes; "
                    f"camera {camera_name!r} has {tuple(image.shape)}, expected {tuple(reference_shape)}."
                )
                raise ValueError(message)
            views.append(image)

        unexpected = set(images) - set(self.camera_names)
        if unexpected:
            message = f"Received unconfigured Patch Policy cameras: {sorted(unexpected)!r}."
            raise ValueError(message)

        # The public input remains one 5D tensor per camera. A view axis exists
        # only internally while all images share one encoder invocation.
        stacked = torch.stack(views, dim=2)  # [B, T, V, C, H, W]
        tokens = self.encoder(stacked)  # [B, T, V, P, E]
        return tokens.flatten(2, 3)  # [B, T, V * P, E]

    def _context(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Encode the complete observation window into visual context.

        Returns:
            Context tokens with temporal and patch dimensions flattened.
        """
        return {"tokens": self._encode_observations(batch).flatten(1, 2)}

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

    def train(self, mode: bool = True) -> PatchPolicyModel:  # noqa: FBT001, FBT002
        """Set module mode while keeping diffusion EMA weights in evaluation mode.

        Returns:
            This model.
        """
        super().train(mode=mode)
        if self.ema_action_head is not None:
            self.ema_action_head.eval()
        return self

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update diffusion inference weights with the upstream EMA schedule."""
        if self.ema_action_head is None:
            return
        step = max(0, int(self.ema_optimization_step) - 1)
        decay = min(1 - (1 + step) ** -0.75, 0.9999) if step > 0 else 0.0
        for averaged, current in zip(
            self.ema_action_head.parameters(),
            self.action_head.parameters(),
            strict=True,
        ):
            averaged.lerp_(current.detach().to(averaged.dtype), 1 - decay)
        self.ema_optimization_step += 1

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the reduced training loss and metrics.

        Args:
            batch: Preprocessed batch containing images and target actions.

        Returns:
            The scalar loss and detached logging metrics.
        """
        valid_steps = in_episode_bound(batch)
        context = self._context(batch)
        if isinstance(self.action_head, BehaviourTransformerHead):
            loss = self.action_head.compute_loss(batch["action"], context, valid_steps=valid_steps)
            return loss, {"loss": loss.detach()}
        losses = self.action_head.compute_loss(batch["action"], context)
        loss = reduce_losses(losses, valid_steps)
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute action mean-squared error for validation.

        Args:
            batch: Preprocessed batch containing images and target actions.

        Returns:
            The scalar validation loss and detached logging metrics.
        """
        sampling_head = self.ema_action_head or self.action_head
        actions = sampling_head.sample(self._context(batch))
        target = batch["action"][:, self._n_obs_steps - 1 : self._n_obs_steps - 1 + self._chunk_size]
        valid_steps = in_episode_bound(batch)
        if valid_steps is not None:
            valid_steps = valid_steps[:, self._n_obs_steps - 1 : self._n_obs_steps - 1 + self._chunk_size]
        loss = reduce_losses(F.mse_loss(actions, target, reduction="none"), valid_steps)
        return loss, {"loss": loss.detach(), "action_mse": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Predict one normalized action chunk.

        Args:
            batch: Preprocessed batch containing images.

        Returns:
            A normalized action chunk for each batch item.
        """
        sampling_head = self.ema_action_head or self.action_head
        return sampling_head.sample(self._context(batch))

    @property
    def action_delta_indices(self) -> list[int]:
        """Dataset offsets required for action chunks.

        Returns:
            Consecutive offsets covering the action horizon.
        """
        return list(range(1 - self._n_obs_steps, self._chunk_size))

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
