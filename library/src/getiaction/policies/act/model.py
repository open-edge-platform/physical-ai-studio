# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""ACT torch model."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain
from typing import cast

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter  # noqa: PLC2701
from torchvision.ops.misc import FrozenBatchNorm2d

from getiaction.data import Feature, FeatureType, Observation
from getiaction.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

from .config import ACTConfig


class ACT(nn.Module):
    """Action Chunking Transformer (ACT) model.

    Supports training and inference modes.
    """

    def __init__(  # noqa: PLR0913
        self,
        action_features: dict[str, Feature],
        observation_features: dict[str, Feature],
        *,
        chunk_size: int = 100,
        n_action_steps: int = 100,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: int = False,
        pre_norm: bool = False,
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        use_vae: bool = True,
        latent_dim: int = 32,
        n_vae_encoder_layers: int = 4,
        temporal_ensemble_coeff: float | None = None,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
    ) -> None:
        """Initialize the ACT model.

        Args:
            action_features (dict[str, Feature]): Dictionary containing action features.
                Must contain exactly one action feature.
            observation_features (dict[str, Feature]): Dictionary containing observation features.
                Must contain exactly one state observation feature and at least one visual observation feature.
            chunk_size (int, optional): Number of actions to predict in a single forward pass. Defaults to 100.
            n_action_steps (int, optional): Number of action steps in the sequence. Defaults to 100.
            vision_backbone (str, optional): Vision backbone architecture to use. Defaults to "resnet18".
            pretrained_backbone_weights (str | None, optional): Pretrained weights for the backbone.
                Defaults to "ResNet18_Weights.IMAGENET1K_V1".
            replace_final_stride_with_dilation (int, optional): Whether to replace final stride with dilation.
                Defaults to False.
            pre_norm (bool, optional): Whether to use pre-normalization in transformer layers. Defaults to False.
            dim_model (int, optional): Model dimension for transformer. Defaults to 512.
            n_heads (int, optional): Number of attention heads in transformer. Defaults to 8.
            dim_feedforward (int, optional): Dimension of feedforward network in transformer. Defaults to 3200.
            feedforward_activation (str, optional): Activation function for feedforward network. Defaults to "relu".
            n_encoder_layers (int, optional): Number of encoder layers in transformer. Defaults to 4.
            n_decoder_layers (int, optional): Number of decoder layers in transformer. Defaults to 1.
            use_vae (bool, optional): Whether to use Variational Autoencoder. Defaults to True.
            latent_dim (int, optional): Dimension of VAE latent space. Defaults to 32.
            n_vae_encoder_layers (int, optional): Number of VAE encoder layers. Defaults to 4.
            temporal_ensemble_coeff (float | None, optional): Coefficient for temporal ensemble.
                Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            kl_weight (float, optional): Weight for KL divergence loss in VAE. Defaults to 10.0.

        Raises:
            ValueError: If the number of state observation features is not exactly one.
            ValueError: If the number of action features is not exactly one.
            ValueError: If no visual observation features are provided.

        Note:
            The ACT model requires:
            - Exactly one state observation feature (FeatureType.STATE)
            - Exactly one action feature
            - At least one visual observation feature (FeatureType.VISUAL)
        """
        super().__init__()

        state_observation_features = [v for v in observation_features.values() if v.ftype == FeatureType.STATE]

        if len(state_observation_features) != 1:
            msg = "ACT model supports exactly one state observation feature."
            raise ValueError(msg)

        if len(action_features) != 1:
            msg = "ACT model supports exactly one action feature."
            raise ValueError(msg)
        action_feature = next(iter(action_features.values()))

        input_features: dict[str | Observation.ComponentKeys, Feature] = {
            Observation.ComponentKeys.STATE: state_observation_features[0],
        }

        visual_observation_features = [v for v in observation_features.values() if v.ftype == FeatureType.VISUAL]

        if len(visual_observation_features) == 1:
            input_features[Observation.ComponentKeys.IMAGES] = visual_observation_features[0]
        elif len(visual_observation_features) > 1:
            for vf in visual_observation_features:
                if vf.name is not None:
                    input_features[vf.name] = vf
        else:
            msg = "ACT model requires at least one visual observation feature."
            raise ValueError(msg)

        output_features: dict[str | Observation.ComponentKeys, Feature] = {
            Observation.ComponentKeys.ACTION: action_feature,
        }

        self._config = _ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
            pre_norm=pre_norm,
            dim_model=dim_model,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            use_vae=use_vae,
            latent_dim=latent_dim,
            n_vae_encoder_layers=n_vae_encoder_layers,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            dropout=dropout,
            kl_weight=kl_weight,
        )

        self._input_normalizer = FeatureNormalizeTransform(
            features=input_features,
            norm_map=self._config.normalization_mapping,
        )
        self._output_denormalizer = FeatureNormalizeTransform(
            output_features,
            self._config.normalization_mapping,
            inverse=True,
        )
        self._model = _ACT(self._config)

    @classmethod
    def from_config(cls, config: ACTConfig) -> "ACT":
        """Create an ACT model from a configuration object.

        Args:
            config (ACTConfig): Configuration object containing model parameters.

        Returns:
            ACT: Initialized ACT model instance.
        """
        return cls(
            action_features=config.output_features,
            observation_features=config.input_features,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            vision_backbone=config.vision_backbone,
            pretrained_backbone_weights=config.pretrained_backbone_weights,
            replace_final_stride_with_dilation=config.replace_final_stride_with_dilation,
            pre_norm=config.pre_norm,
            dim_model=config.dim_model,
            n_heads=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            feedforward_activation=config.feedforward_activation,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            use_vae=config.use_vae,
            latent_dim=config.latent_dim,
            n_vae_encoder_layers=config.n_vae_encoder_layers,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
            dropout=config.dropout,
            kl_weight=config.kl_weight,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        """Forward pass through the ACT model.

        In training mode, computes loss components including L1 loss and optional KL divergence loss
        for VAE regularization. In evaluation mode, predicts action chunks.

        Args:
            batch: Dictionary containing batch data with keys:
                - Observation.ComponentKeys.ACTION: Ground truth actions
                - Observation.ComponentKeys.IMAGES: Input images (dict or tensor)
                - Observation.ComponentKeys.EXTRA: Extra data including action padding mask

        Returns:
            tuple[torch.Tensor, dict[str, float]] | torch.Tensor: In training mode, returns tuple
                of (total_loss, loss_dict) where loss_dict contains 'l1_loss' and optionally 'kld_loss' items.
                In evaluation mode, returns predicted action tensor from predict_action_chunk().

        Note:
            - Input normalization is applied in training mode
            - KL divergence loss is computed when config.use_vae is True
        """
        if self._model.training:
            batch = self._input_normalizer(batch)
            if self._config.image_features:
                batch_ = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                if isinstance(batch[Observation.ComponentKeys.IMAGES], dict):
                    batch_[Observation.ComponentKeys.IMAGES] = [
                        batch[Observation.ComponentKeys.IMAGES][key] for key in self._config.image_features
                    ]
                    batch = batch_
                else:
                    batch[Observation.ComponentKeys.IMAGES] = [
                        batch[Observation.ComponentKeys.IMAGES],
                    ]
            actions_hat, (mu_hat, log_sigma_x2_hat) = self._model(batch)

            l1_loss = (
                F.l1_loss(batch[Observation.ComponentKeys.ACTION], actions_hat, reduction="none")
                * ~batch[Observation.ComponentKeys.EXTRA]["action_is_pad"].unsqueeze(-1)
            ).mean()

            loss_dict = {"l1_loss": l1_loss.item()}
            if self._config.use_vae:
                # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
                # each dimension independently, we sum over the latent dimension to get the total
                # KL-divergence per batch element, then take the mean over the batch.
                # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
                mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
                loss_dict["kld_loss"] = mean_kld.item()
                loss = l1_loss + mean_kld * self._config.kl_weight
            else:
                loss = l1_loss

            return loss, loss_dict
        return self.predict_action_chunk(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicts a chunk of actions from a batch of observations.

        This method processes a batch of input data through the model to generate
        corresponding actions. It normalizes inputs, handles image features if configured,
        runs the model inference, and denormalizes the output actions.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing observation data
                with string keys and tensor values. Expected to contain various
                observation components that the model requires for prediction.

        Returns:
            torch.Tensor: A tensor containing the predicted actions.
                The tensor shape and content depend on the model's action space configuration.

        Note:
            - The model is set to evaluation mode during prediction
            - Input normalization is applied to the batch
        """
        self.eval()

        batch = self._input_normalizer(batch)
        if self._config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[Observation.ComponentKeys.IMAGES] = [batch[key] for key in self._config.image_features]

        actions = self._model(batch)[0]  # only select the actions, ignore the latent params
        return self._output_denormalizer({Observation.ComponentKeys.ACTION: actions})[Observation.ComponentKeys.ACTION]

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            list[int]: A list of relative action indices.
        """
        return list(range(self._config.chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """Get indices of observations relative to the current timestep.

        Returns:
            list[int]: A list of relative observation indices.
        """
        return None


@dataclass(frozen=True)
class _ACTConfig(ACTConfig):
    normalization_mapping: dict[FeatureType, NormalizationType] = field(
        default_factory=lambda: {
            FeatureType.VISUAL: NormalizationType.MEAN_STD,
            FeatureType.STATE: NormalizationType.MEAN_STD,
            FeatureType.ACTION: NormalizationType.MEAN_STD,
        },
    )

    def __post_init__(self) -> None:
        """Post-initialization validation for ACT model configuration.

        Validates the configuration parameters after dataclass initialization to ensure:
        - Vision backbone is a ResNet variant
        - Temporal ensemble coefficient is only used with single action steps
        - Number of action steps doesn't exceed chunk size
        - Only single observation steps are supported

        Raises:
            ValueError: If vision_backbone is not a ResNet variant, if n_action_steps
                exceeds chunk_size, or if n_obs_steps is not 1.
            NotImplementedError: If temporal_ensemble_coeff is used with n_action_steps > 1.
        """
        if not self.vision_backbone.startswith("resnet"):
            msg = f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            raise ValueError(
                msg,
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            msg = (
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "This is because the policy needs to be queried every step to compute the ensembled action."
            )
            raise NotImplementedError(msg)
        if self.n_action_steps > self.chunk_size:
            msg = (
                "The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
            raise ValueError(
                msg,
            )
        if self.n_obs_steps != 1:
            msg = f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            raise ValueError(
                msg,
            )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            msg = "You must provide at least one image or the environment state among the inputs."
            raise ValueError(msg)

    @property
    def robot_state_feature(self) -> Feature | None:
        for ft_name, ft in self.input_features.items():
            if ft.ftype is FeatureType.STATE and ft_name == Observation.ComponentKeys.STATE:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, Feature]:
        return {key: ft for key, ft in self.input_features.items() if ft.ftype is FeatureType.VISUAL}

    @property
    def env_state_feature(self) -> Feature | None:
        for ft in self.input_features.values():
            if ft.ftype is FeatureType.ENV:
                return ft
        return None

    @property
    def action_feature(self) -> Feature:
        for ft_name, ft in self.output_features.items():
            if ft.ftype is FeatureType.ACTION and ft_name == Observation.ComponentKeys.ACTION:
                return ft
        msg = "No action feature found in output features."
        raise ValueError(msg)


class _ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy.

    Note: In this code we use the terms `vae_encoder`, 'encoder', `decoder`. The meanings are as follows.
        - The `vae_encoder` is, as per the literature around variational auto-encoders (VAE), the part of the
          model that encodes the target data (a sequence of actions), and the condition (the robot
          joint-space).
        - A transformer with an `encoder` (not the VAE encoder) and `decoder` (not the VAE decoder) with
          cross-attention is used as the VAE decoder. For these terms, we drop the `vae_` prefix because we
          have an option to train this model without the variational objective (in which case we drop the
          `vae_encoder` altogether, and nothing about this model has anything to do with a VAE).

                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    """

    def __init__(self, config: _ACTConfig) -> None:
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = _ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    cast("tuple", self.config.robot_state_feature.shape)[0],
                    config.dim_model,
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                cast("tuple", self.config.action_feature.shape)[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                _create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = _ACTEncoder(config)
        self.decoder = _ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                cast("tuple", self.config.robot_state_feature.shape)[0],
                config.dim_model,
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                cast("tuple", self.config.env_state_feature.shape)[0],
                config.dim_model,
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features,
                config.dim_model,
                kernel_size=1,
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = _ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, cast("tuple", self.config.action_feature.shape)[0])

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:  # noqa: PLR0914, PLR0915
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.

        Raises:
            RuntimeError: If action features are missing in input batch in VAE mode.
        """
        if self.config.use_vae and self.training and "action" not in batch:
            msg = "Actions must be provided when using the variational objective in training mode."
            raise RuntimeError(msg)

        if "images" in batch:
            batch_size = batch[Observation.ComponentKeys.IMAGES][0].shape[0]
        else:
            batch_size = batch["environment_state"].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and "action" in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight,
                "1 d -> b 1 d",
                b=batch_size,
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[Observation.ComponentKeys.STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[Observation.ComponentKeys.ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                fill_value=False,
                device=batch[Observation.ComponentKeys.STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch[Observation.ComponentKeys.EXTRA]["action_is_pad"]],
                axis=1,
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # (rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[Observation.ComponentKeys.STATE].device,
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[Observation.ComponentKeys.STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(
                self.encoder_env_state_input_proj(batch["environment_state"]),
            )

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[Observation.ComponentKeys.IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class _ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: _ACTConfig, *, is_vae_encoder: bool = False) -> None:
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([_ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class _ACTEncoderLayer(nn.Module):
    def __init__(self, config: _ACTConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = _get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class _ACTDecoder(nn.Module):
    def __init__(self, config: _ACTConfig) -> None:
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([_ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class _ACTDecoderLayer(nn.Module):
    def __init__(self, config: _ACTConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = _get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    @staticmethod
    def maybe_add_pos_embed(tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the transformer decoder layer.

        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).

        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def _create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> torch.Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
        dimension: Dimensionality of the position embeddings.

    Returns:
        (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class _ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int) -> None:
        """Initialize the positional encoding layer.

        Args:
            dimension (int): The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to generate 2D sinusoidal positional embeddings.

        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.

        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        return torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)


def _get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string.

    Args:
        activation (str): Name of the activation function. Supported values are:
            - "relu": Returns F.relu function
            - "gelu": Returns F.gelu function
            - "glu": Returns F.glu function
    Returns:
        Callable: The corresponding PyTorch activation function.

    Raises:
        RuntimeError: If the activation function name is not supported.

    Example:
        >>> activation_fn = _get_activation_fn("relu")
        >>> output = activation_fn(input_tensor)
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    msg = f"Unknown activation function: {activation}"
    raise RuntimeError(msg)


__all__ = ["ACT"]
