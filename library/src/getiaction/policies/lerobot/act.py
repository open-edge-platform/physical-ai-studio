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
from getiaction.policies.lerobot.mixin import LeRobotFromConfig

if TYPE_CHECKING:
    from torch import nn

    from getiaction.data import Observation
    from getiaction.data.observation import GymObservation

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


class ACT(Policy, LeRobotFromConfig):
    """Action Chunking Transformer (ACT) policy from LeRobot.

    PyTorch Lightning wrapper around LeRobot's ACT implementation that provides
    flexible configuration options and seamless integration with the getiaction
    data pipeline. The policy supports lazy initialization, where the LeRobot
    policy is created during setup() after dataset features are loaded.

    ACT uses a transformer-based architecture with optional VAE encoding to predict
    sequences of actions (chunks) from visual observations. It's particularly
    effective for manipulation tasks requiring temporal consistency.

    The wrapper supports multiple configuration methods through the ``LeRobotFromConfig`` mixin.
    See ``LeRobotFromConfig`` for detailed configuration examples.

    Examples:
        Basic usage with explicit arguments (recommended):
            >>> from getiaction.policies.lerobot import ACT
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> from lightning import Trainer

            >>> # Create policy with explicit parameters
            >>> policy = ACT(
            ...     dim_model=512,
            ...     chunk_size=100,
            ...     n_action_steps=100,
            ...     use_vae=True,
            ...     n_encoder_layers=4,
            ... )

            >>> # Create datamodule
            >>> datamodule = LeRobotDataModule(
            ...     repo_id="lerobot/pusht",
            ...     train_batch_size=8,
            ... )

            >>> # Train
            >>> trainer = Trainer(max_epochs=100)
            >>> trainer.fit(policy, datamodule)

        Using kwargs for unlisted parameters:
            >>> policy = ACT(
            ...     dim_model=512,
            ...     chunk_size=100,
            ...     feedforward_activation="gelu",  # Via kwargs
            ...     pre_norm=True,  # Via kwargs
            ... )

        Using configuration file (alternative):
            >>> # From dict, YAML, Pydantic, or LeRobot config
            >>> policy = ACT.from_config("config.yaml")
            >>> # or
            >>> policy = ACT.from_config({"dim_model": 512, "chunk_size": 100})

        Using pre-configured LeRobot config (advanced):
            >>> from lerobot.policies.act.configuration_act import ACTConfig as LeRobotACTConfig
            >>> lerobot_config = LeRobotACTConfig(dim_model=512, chunk_size=100)
            >>> policy = ACT.from_config(lerobot_config)

        YAML configuration with LightningCLI (explicit args):

            ```yaml
            # config.yaml
            model:
              class_path: getiaction.policies.lerobot.ACT
              init_args:
                dim_model: 512
                chunk_size: 100
                n_action_steps: 100
                use_vae: true
            data:
              class_path: getiaction.data.lerobot.LeRobotDataModule
              init_args:
                repo_id: lerobot/pusht
                train_batch_size: 8
            ```

            Command line usage:

            ```bash
            getiaction fit --config config.yaml
            ```

        CLI overrides (with explicit args in YAML):

            ```bash
            getiaction fit --config config.yaml --model.dim_model 1024
            ```

    Note:
        The policy is initialized lazily during the setup() phase, which is called
        automatically by Lightning before training. During setup, input/output features
        are extracted from the dataset and used to configure the underlying LeRobot policy.

    Note:
        The ``LeRobotFromConfig`` mixin provides multiple configuration methods:
        ``from_config()``, ``from_dict()``, ``from_yaml()``, ``from_pydantic()``,
        ``from_dataclass()``, and ``from_lerobot_config()``. See ``LeRobotFromConfig``
        documentation for detailed usage examples.

    See Also:
        - LeRobotDataModule: For loading LeRobot datasets
        - LeRobotFromConfig: Configuration mixin with detailed examples
        - FormatConverter: For format conversion between getiaction and lerobot
        - lerobot.policies.act.modeling_act.ACTPolicy: Underlying LeRobot implementation
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
        optimizer_lr_backbone: float = 1e-5,
        # Transformer
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        # VAE
        use_vae: bool = True,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        # Regularization
        dropout: float = 0.1,
        # Optimizer
        learning_rate: float = 1e-5,
        # Inference
        temporal_ensemble_coeff: float | None = None,
        # Additional parameters via kwargs
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize ACT policy wrapper.

        The LeRobot policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        For loading from dicts, YAML, Pydantic models, or LeRobot configs, use the
        inherited methods from LeRobotFromConfig mixin: ``from_config()``,
        ``from_dict()``, ``from_yaml()``, ``from_pydantic()``, ``from_dataclass()``,
        or ``from_lerobot_config()``.

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
            **kwargs: Additional ACTConfig parameters (e.g., feedforward_activation, pre_norm).

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "ACT requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__()

        # Build config dict from explicit args
        self._config_object = None
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
        self._lerobot_policy: _LeRobotACTPolicy
        self.model: nn.Module | None = None

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> _LeRobotACTPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot ACT policy.

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if not hasattr(self, "_lerobot_policy") or self._lerobot_policy is None:
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

        if hasattr(self, "_lerobot_policy") and self._lerobot_policy is not None:
            return  # Already initialized

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
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

        # Create or update LeRobot ACT configuration based on what user provided
        if self._config_object is not None:
            # User provided a full config object - update input/output features
            lerobot_config = self._config_object
            lerobot_config.input_features = features
            lerobot_config.output_features = features
        else:
            # User provided dict or explicit args - create config
            lerobot_config = _LeRobotACTConfig(  # type: ignore[misc]
                input_features=features,
                output_features=features,
                **self._config_kwargs,  # type: ignore[arg-type]
            )

        # Initialize the policy
        policy = _LeRobotACTPolicy(lerobot_config, dataset_stats=stats)  # type: ignore[arg-type,misc]
        self.add_module("_lerobot_policy", policy)
        self.model = self._lerobot_policy.model

    def forward(
        self,
        batch: dict[str, torch.Tensor] | Observation,
        *args: Any,  # noqa: ARG002, ANN401
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> torch.Tensor:
        """Forward pass for ACT policy.

        Args:
            batch: A batch of data containing observations and actions (dict or Observation).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The action predictions from the policy.
        """
        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        actions, _ = self.lerobot_policy.model(batch_dict)
        return actions

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's custom parameter groups.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        return torch.optim.AdamW(self.lerobot_policy.get_optim_params(), lr=self.learning_rate)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: A batch of data containing observations and actions.
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        total_loss, loss_dict = self.lerobot_policy.forward(batch_dict)
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=False)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: GymObservation, batch_idx: int) -> dict[str, float]:
        """Validation step of the policy.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader only returns GymObservation batches.

        Args:
            batch: GymObservation containing the environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: GymObservation, batch_idx: int) -> dict[str, float]:
        """Test step of the policy.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader only returns GymObservation batches.

        Args:
            batch: GymObservation containing the environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def select_action(self, batch: dict[str, torch.Tensor] | Observation) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Args:
            batch: A batch of data containing observations (dict or Observation).

        Returns:
            The selected action tensor.
        """
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        # TODO (samet-akcay): Manual device handling required for gym rollouts.  # noqa: FIX002
        # https://github.com/open-edge-platform/geti-action/issues/57
        #
        # During gym rollouts, observations come directly from env.step() as CPU numpy arrays,
        # bypassing Lightning's transfer_batch_to_device hook. This device transfer ensures
        # compatibility with GPU training. Future improvement: move this to base Policy class
        # or rollout function for cleaner separation of concerns.
        device = next(self.lerobot_policy.parameters()).device
        batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}

        return self.lerobot_policy.select_action(batch_dict)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
