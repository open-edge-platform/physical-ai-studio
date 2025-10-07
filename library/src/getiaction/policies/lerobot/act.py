# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action Chunking Transformer (ACT) policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's ACT implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning_utilities.core.imports import module_available

from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.policies.base import Policy

if TYPE_CHECKING:
    from torch import nn

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig as _LeRobotACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy as _LeRobotACTPolicy

    LEROBOT_AVAILABLE = True
else:
    LeRobotDataset = None
    dataset_to_policy_features = None
    _LeRobotACTConfig = None
    _LeRobotACTPolicy = None
    LEROBOT_AVAILABLE = False


class ACT(Policy):
    """Action Chunking Transformer from LeRobot with lazy initialization.

    The LeRobot policy is created in setup() after the dataset is loaded.
    This enables YAML configuration while maintaining compatibility with Python scripts.

    Example YAML usage:
        model:
          class_path: getiaction.policies.lerobot.ACT
          init_args:
            dim_model: 512
            chunk_size: 100
        data:
          class_path: getiaction.data.lerobot.LeRobotDataModule
          init_args:
            repo_id: lerobot/pusht

    Example Python usage:
        policy = ACT(dim_model=512, chunk_size=100)
        datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
        trainer.fit(policy, datamodule)  # setup() called automatically
    """

    def __init__(  # noqa: PLR0913
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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize ACT policy wrapper.

        The LeRobot policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        Args:
            dim_model: Transformer model dimension.
            chunk_size: Number of action predictions per forward pass.
            n_action_steps: Number of action steps to execute.
            optimizer_lr_backbone: Learning rate for vision backbone.
            vision_backbone: Vision encoder architecture (e.g., "resnet18").
            pretrained_backbone_weights: Pretrained weights for vision backbone.
            use_vae: Whether to use VAE for action encoding.
            latent_dim: Dimension of VAE latent space.
            dropout: Dropout probability.
            kl_weight: Weight for KL divergence loss.
            n_encoder_layers: Number of transformer encoder layers.
            n_decoder_layers: Number of transformer decoder layers.
            n_heads: Number of attention heads.
            dim_feedforward: Dimension of feedforward network.
            learning_rate: Learning rate for optimizer.
            temporal_ensemble_coeff: Coefficient for temporal ensembling.
            **kwargs: Additional arguments passed to ACTConfig.

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "ACT requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__()

        # Store configuration for use in setup()
        self._config_kwargs = {
            "dim_model": dim_model,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "optimizer_lr_backbone": optimizer_lr_backbone,
            "vision_backbone": vision_backbone,
            "pretrained_backbone_weights": pretrained_backbone_weights,
            "use_vae": use_vae,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "kl_weight": kl_weight,
            "n_encoder_layers": n_encoder_layers,
            "n_decoder_layers": n_decoder_layers,
            "n_heads": n_heads,
            "dim_feedforward": dim_feedforward,
            "temporal_ensemble_coeff": temporal_ensemble_coeff,
            **kwargs,
        }

        self.learning_rate = learning_rate
        self._framework = "lerobot"

        # Policy will be initialized in setup()
        self._lerobot_policy: _LeRobotACTPolicy | None = None
        self.model: nn.Module | None = None
        self._framework_policy: _LeRobotACTPolicy | None = None

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> _LeRobotACTPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot ACT policy.

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if self._lerobot_policy is None:
            msg = "Policy not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._lerobot_policy

    def setup(self, stage: str) -> None:
        """Set up the policy from datamodule if not already initialized.

        This method is called by Lightning before fit/validate/test/predict.
        It extracts features from the datamodule's training dataset and
        initializes the policy if it wasn't already created in __init__.

        Args:
            stage: The stage of training ('fit', 'validate', 'test', or 'predict')

        Raises:
            TypeError: If the train_dataset is not a LeRobot dataset.
        """
        del stage  # Unused argument

        if self._lerobot_policy is not None:
            return  # Already initialized

        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both data formats
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            # Wrapped in adapter for getiaction format conversion
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif LeRobotDataset is not None and isinstance(train_dataset, LeRobotDataset):
            # Dataset is raw LeRobotDataset (data_format="lerobot")
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule with appropriate data_format."
            )
            raise TypeError(msg)
        features = dataset_to_policy_features(lerobot_dataset.meta.features)
        stats = lerobot_dataset.meta.stats

        # Create LeRobot ACT configuration
        lerobot_config = _LeRobotACTConfig(  # type: ignore[misc]
            input_features=features,
            output_features=features,
            **self._config_kwargs,
        )

        # Initialize the policy
        self._lerobot_policy = _LeRobotACTPolicy(lerobot_config, dataset_stats=stats)  # type: ignore[arg-type,misc]
        self.model = self._lerobot_policy.model
        self._framework_policy = self._lerobot_policy

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass delegates to LeRobot.

        Args:
            batch: A batch of data containing observations.

        Returns:
            The action predictions from the policy.
        """
        actions, _ = self.lerobot_policy.model(batch)
        return actions

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: A batch of data containing observations and actions.
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch = FormatConverter.to_lerobot_dict(batch)

        total_loss, loss_dict = self.lerobot_policy.forward(batch)
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=False)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step of the policy.

        Args:
            batch: A batch of data containing observations and actions.
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch = FormatConverter.to_lerobot_dict(batch)

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
        """Select action (inference mode) through LeRobot.

        Args:
            batch: A batch of data containing observations.

        Returns:
            The selected action tensor.
        """
        return self.lerobot_policy.select_action(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's custom parameter groups.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        return torch.optim.AdamW(self.lerobot_policy.get_optim_params(), lr=self.learning_rate)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
