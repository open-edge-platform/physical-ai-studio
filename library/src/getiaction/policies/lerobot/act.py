# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action Chunking Transformer (ACT) policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's ACT implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from getiaction.policies.base import Policy

if TYPE_CHECKING:
    from torch import nn

try:
    from lerobot.policies.act.configuration_act import ACTConfig as _LeRobotACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy as _LeRobotACTPolicy

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    _LeRobotACTPolicy = None
    _LeRobotACTConfig = None


class ACT(Policy):
    """Action Chunking Transformer from LeRobot."""

    def __init__(
        self,
        *,
        dim_model: int = 512,
        chunk_size: int = 100,
        n_action_steps: int = 100,
        optimizer_lr_backbone: float = 1e-5,
        vision_backbone: str = "resnet18",
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1",
        use_vae: bool = True,
        latent_dim: int = 32,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        learning_rate: float = 1e-5,
        temporal_ensemble_coeff: float | None = None,
        stats: dict[str, dict[str, torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ACT policy wrapper."""
        if not LEROBOT_AVAILABLE:
            msg = "ACT requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__()

        lerobot_config = _LeRobotACTConfig(
            dim_model=dim_model,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            optimizer_lr_backbone=optimizer_lr_backbone,
            vision_backbone=vision_backbone,
            pretrained_backbone_weights=pretrained_backbone_weights,
            use_vae=use_vae,
            latent_dim=latent_dim,
            dropout=dropout,
            kl_weight=kl_weight,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            **kwargs,
        )

        self.lerobot_policy = _LeRobotACTPolicy(lerobot_config, dataset_stats=stats)
        self.model: nn.Module = self.lerobot_policy.model
        self.learning_rate = learning_rate
        self.stats = stats
        self._framework = "lerobot"
        self._framework_policy = self.lerobot_policy
        self.save_hyperparameters()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass delegates to LeRobot."""
        actions, _ = self.lerobot_policy.model(batch)
        return actions

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation."""
        total_loss, loss_dict = self.lerobot_policy.forward(batch)
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=False)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Workaround for LeRobot bug: VAE fails in eval mode
        was_training = self.training
        if self.lerobot_policy.config.use_vae and not was_training:
            self.train()
        total_loss, loss_dict = self.lerobot_policy.forward(batch)
        if not was_training:
            self.eval()
        for key, value in loss_dict.items():
            self.log(f"val/{key}", value, prog_bar=False)
        self.log("val/loss", total_loss, prog_bar=True)
        return total_loss

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action (inference mode) through LeRobot."""
        return self.lerobot_policy.select_action(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's custom parameter groups."""
        return torch.optim.AdamW(self.lerobot_policy.get_optim_params(), lr=self.learning_rate)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
