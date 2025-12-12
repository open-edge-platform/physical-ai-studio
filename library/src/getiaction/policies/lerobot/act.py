# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action Chunking Transformer (ACT) policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's ACT implementation
with explicit typed parameters for better IDE support and documentation.
"""

from __future__ import annotations

from typing import Any

from lightning_utilities.core.imports import module_available

from getiaction.policies.lerobot.universal import LeRobotPolicy

LEROBOT_AVAILABLE = bool(module_available("lerobot"))


class ACT(LeRobotPolicy):
    """Action Chunking Transformer (ACT) policy from LeRobot.

    PyTorch Lightning wrapper around LeRobot's ACT implementation that provides
    explicit typed parameters for better IDE support. Inherits all functionality
    from LeRobotPolicy.

    ACT uses a transformer-based architecture with optional VAE encoding to predict
    sequences of actions (chunks) from visual observations. It's particularly
    effective for manipulation tasks requiring temporal consistency.

    Examples:
        Load pretrained model from HuggingFace Hub:

            >>> from getiaction.policies.lerobot import ACT
            >>> policy = ACT.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

        Create from dataset (eager initialization):

            >>> policy = ACT.from_dataset(
            ...     "lerobot/pusht",
            ...     dim_model=512,
            ...     chunk_size=100,
            ... )

        Train from scratch with explicit arguments:

            >>> from getiaction.policies.lerobot import ACT
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> from getiaction.train import Trainer

            >>> policy = ACT(
            ...     dim_model=512,
            ...     chunk_size=100,
            ...     n_action_steps=100,
            ...     use_vae=True,
            ... )

            >>> datamodule = LeRobotDataModule(
            ...     repo_id="lerobot/pusht",
            ...     train_batch_size=8,
            ... )

            >>> trainer = Trainer(max_epochs=100)
            >>> trainer.fit(policy, datamodule)

        YAML configuration with LightningCLI:

            ```yaml
            model:
              class_path: getiaction.policies.lerobot.ACT
              init_args:
                dim_model: 512
                chunk_size: 100
                n_action_steps: 100
                use_vae: true
            ```

    Note:
        This class provides explicit typed parameters for IDE autocomplete.
        For dynamic policy selection, use LeRobotPolicy directly.

    See Also:
        - LeRobotPolicy: Universal wrapper for any LeRobot policy
        - LeRobotDataModule: For loading LeRobot datasets
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Architecture
        dim_model: int = 512,
        chunk_size: int = 100,
        n_action_steps: int = 100,
        # Vision backbone
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation: bool = False,
        # Transformer
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_vae_encoder_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        feedforward_activation: str = "relu",
        pre_norm: bool = False,
        # VAE
        use_vae: bool = True,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        # Regularization
        dropout: float = 0.1,
        # Optimizer
        optimizer_lr: float = 1e-5,
        optimizer_lr_backbone: float = 1e-5,
        optimizer_weight_decay: float = 1e-4,
        # Inference
        n_obs_steps: int = 1,
        temporal_ensemble_coeff: float | None = None,
        # Additional parameters via kwargs
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize ACT policy wrapper.

        Args:
            dim_model: Transformer model dimension.
            chunk_size: Number of action predictions per forward pass.
            n_action_steps: Number of action steps to execute.
            vision_backbone: Vision encoder architecture (e.g., "resnet18").
            pretrained_backbone_weights: Pretrained weights for vision backbone.
            replace_final_stride_with_dilation: Whether to replace final stride with dilation.
            n_encoder_layers: Number of transformer encoder layers.
            n_decoder_layers: Number of transformer decoder layers.
            n_vae_encoder_layers: Number of VAE encoder layers.
            n_heads: Number of attention heads.
            dim_feedforward: Dimension of feedforward network.
            feedforward_activation: Activation function for feedforward layers.
            pre_norm: Whether to use pre-normalization in transformer.
            use_vae: Whether to use VAE for action encoding.
            latent_dim: Dimension of VAE latent space.
            kl_weight: Weight for KL divergence loss.
            dropout: Dropout probability.
            optimizer_lr: Learning rate for optimizer.
            optimizer_lr_backbone: Learning rate for vision backbone.
            optimizer_weight_decay: Weight decay for optimizer.
            n_obs_steps: Number of observation steps.
            temporal_ensemble_coeff: Coefficient for temporal ensembling.
            **kwargs: Additional ACTConfig parameters.

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "ACT requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__(
            policy_name="act",
            dim_model=dim_model,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            replace_final_stride_with_dilation=replace_final_stride_with_dilation,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_vae_encoder_layers=n_vae_encoder_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            feedforward_activation=feedforward_activation,
            pre_norm=pre_norm,
            use_vae=use_vae,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            dropout=dropout,
            optimizer_lr=optimizer_lr,
            optimizer_lr_backbone=optimizer_lr_backbone,
            optimizer_weight_decay=optimizer_weight_decay,
            n_obs_steps=n_obs_steps,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            **kwargs,
        )
