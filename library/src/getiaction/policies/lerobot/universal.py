#!/usr/bin/env python
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Universal wrapper for any LeRobot policy.

This module provides a generic wrapper that can instantiate any LeRobot policy
dynamically without requiring explicit wrappers for each policy type.

For users who prefer explicit wrappers with full parameter definitions and IDE
support, see the specific policy modules (e.g., ACT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import LightningModule

from getiaction.policies.base import Policy

if TYPE_CHECKING:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import PolicyFeature

try:
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.policies.pretrained import PreTrainedPolicy

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class LeRobotPolicy(Policy):
    """Universal wrapper for any LeRobot policy.

    This wrapper provides a generic interface to instantiate and use any LeRobot
    policy without requiring an explicit wrapper for each policy type. It's ideal
    for users who want flexibility and don't need full IDE autocomplete support.

    For users who prefer explicit parameter definitions and full IDE support,
    use the specific policy wrappers (e.g., ACT, Diffusion).

    Supported policies:
        - act: Action Chunking Transformer
        - diffusion: Diffusion Policy
        - vqbet: VQ-BeT (VQ-VAE Behavior Transformer)
        - tdmpc: TD-MPC (Temporal Difference Model Predictive Control)
        - sac: Soft Actor-Critic
        - pi0: Vision-Language Policy
        - pi05: PI0.5 (Improved PI0)
        - pi0fast: Fast Inference PI0
        - smolvla: Small Vision-Language-Action

    Example:
        >>> # Option 1: Pass config as dict
        >>> policy = LeRobotPolicy(
        ...     policy_name="diffusion",
        ...     input_features=features,
        ...     output_features=features,
        ...     num_steps=100,
        ...     noise_scheduler="ddpm",
        ...     stats=dataset.meta.stats,
        ... )
        >>>
        >>> # Option 2: Pass pre-built config
        >>> from lerobot.policies import DiffusionConfig
        >>> config = DiffusionConfig(
        ...     input_features=features,
        ...     output_features=features,
        ...     num_steps=100,
        ... )
        >>> policy = LeRobotPolicy(
        ...     policy_name="diffusion",
        ...     config=config,
        ...     stats=dataset.meta.stats,
        ... )
        >>>
        >>> # Option 3: Use with LightningCLI
        >>> # In config.yaml:
        >>> # model:
        >>> #   class_path: getiaction.policies.lerobot.LeRobotPolicy
        >>> #   init_args:
        >>> #     policy_name: vqbet
        >>> #     num_clusters: 256
        >>> #     embedding_dim: 64

    Args:
        policy_name: Name of the LeRobot policy to instantiate. See SUPPORTED_POLICIES.
        input_features: Dictionary of input feature definitions (PolicyFeature objects).
        output_features: Dictionary of output feature definitions (PolicyFeature objects).
        config: Pre-built LeRobot config object. If provided, other config kwargs are ignored.
        stats: Dataset statistics for normalization. If provided, will be passed to
               the LeRobot policy constructor.
        learning_rate: Learning rate for optimizer (default: 1e-4).
        **config_kwargs: Additional configuration parameters specific to the policy type.
                        These are passed to the LeRobot config constructor.

    Raises:
        ImportError: If LeRobot is not installed.
        ValueError: If policy_name is not supported.

    Note:
        This is the "universal wrapper" approach. For explicit parameter definitions
        and better IDE support, consider using specific wrappers (e.g., ACT).
    """

    SUPPORTED_POLICIES = [
        "act",
        "diffusion",
        "vqbet",
        "tdmpc",
        "sac",
        "pi0",
        "pi05",
        "pi0fast",
        "smolvla",
    ]

    def __init__(
        self,
        policy_name: str,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        config: PreTrainedConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        learning_rate: float = 1e-4,
        config_kwargs: dict[str, Any] | None = None,
        **extra_config_kwargs: Any,
    ) -> None:
        """Initialize the universal LeRobot policy wrapper.

        Supports both eager initialization (when input_features provided) and lazy
        initialization (features extracted in setup() hook from DataModule).

        Args:
            policy_name: Name of the policy ('diffusion', 'act', 'vqbet', etc.)
            input_features: Optional input feature definitions (lazy if None)
            output_features: Optional output feature definitions (lazy if None)
            config: Pre-built LeRobot config object (optional)
            dataset_stats: Dataset statistics for normalization (optional)
            learning_rate: Learning rate for optimizer
            config_kwargs: Policy-specific parameters as a dict (for YAML configs).
                          See LeRobot's policy config classes for available parameters.
            **extra_config_kwargs: Policy-specific parameters as kwargs (for Python usage).
                                  These are merged with config_kwargs.

        Note:
            When using Lightning CLI/YAML, use `config_kwargs` (nested dict) due to
            CLI validation limitations. When using Python directly, you can use either
            `config_kwargs={}` or pass parameters as **kwargs.
        """
        if not LEROBOT_AVAILABLE:
            msg = (
                "LeRobotPolicy requires LeRobot framework.\n\n"
                "Install with:\n"
                "    pip install lerobot\n\n"
                "Or install getiaction with LeRobot support:\n"
                "    pip install getiaction[lerobot]\n\n"
                "For more information, see: https://github.com/huggingface/lerobot"
            )
            raise ImportError(msg)

        if policy_name not in self.SUPPORTED_POLICIES:
            msg = (
                f"Policy '{policy_name}' is not supported.\n\n"
                f"Supported policies: {', '.join(self.SUPPORTED_POLICIES)}\n\n"
                f"If you need a different policy, either:\n"
                f"  1. Add it to SUPPORTED_POLICIES if it's in LeRobot\n"
                f"  2. Create an explicit wrapper for better IDE support\n"
                f"  3. Use the LeRobot policy directly"
            )
            raise ValueError(msg)

        super().__init__()

        # Store metadata
        self.policy_name = policy_name
        self.learning_rate = learning_rate

        # Merge config_kwargs (from YAML) with extra_config_kwargs (from Python **kwargs)
        merged_config_kwargs = {**(config_kwargs or {}), **extra_config_kwargs}

        # Store for lazy initialization
        self._input_features = input_features
        self._output_features = output_features
        self._provided_config = config
        self._dataset_stats = dataset_stats
        self._config_kwargs = merged_config_kwargs

        # Will be initialized in setup() if not provided
        self.lerobot_policy: PreTrainedPolicy | None = None
        self._config: PreTrainedConfig | None = None

        # If features are provided, initialize immediately (backward compatibility)
        if input_features is not None and output_features is not None:
            self._initialize_policy(input_features, output_features, config, dataset_stats)
        elif config is not None:
            # Config provided directly - can initialize now
            self._initialize_policy(None, None, config, dataset_stats)

        self.save_hyperparameters()

    def _initialize_policy(
        self,
        input_features: dict[str, PolicyFeature] | None,
        output_features: dict[str, PolicyFeature] | None,
        config: PreTrainedConfig | None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    ) -> None:
        """Initialize the LeRobot policy instance.

        Args:
            input_features: Input feature definitions.
            output_features: Output feature definitions.
            config: Pre-built config object.
            dataset_stats: Dataset statistics for normalization.
        """
        # Build or use provided config
        if config is None:
            if input_features is None or output_features is None:
                msg = (
                    "Either 'config' must be provided, or both 'input_features' and 'output_features' must be provided."
                )
                raise ValueError(msg)

            # Remove dataset_stats from config_kwargs if present
            # (it should be passed to policy constructor, not config)
            clean_config_kwargs = {k: v for k, v in self._config_kwargs.items() if k != "dataset_stats"}

            # Create config dynamically using LeRobot's factory
            config = make_policy_config(
                self.policy_name,
                input_features=input_features,
                output_features=output_features,
                **clean_config_kwargs,
            )

        # Get the policy class dynamically
        policy_cls = get_policy_class(self.policy_name)

        # Instantiate the LeRobot policy
        self.lerobot_policy = policy_cls(config, dataset_stats=dataset_stats)

        # Expose the underlying model for Lightning compatibility (if available)
        # Some policies (like Diffusion) don't have a .model attribute
        if hasattr(self.lerobot_policy, "model"):
            self.model = self.lerobot_policy.model

        # Expose framework info
        self._framework = "lerobot"
        self._framework_policy = self.lerobot_policy
        self._config = config

    def setup(self, stage: str) -> None:
        """Lightning hook called before training/validation/test.

        Extracts input/output features from the DataModule's dataset if not provided
        during initialization. This enables YAML-based configuration without requiring
        features to be specified in advance.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if self.lerobot_policy is not None:
            # Already initialized
            return

        # Lazy initialization: extract features from DataModule
        if not hasattr(self.trainer, "datamodule"):
            msg = (
                "Lazy initialization requires a DataModule with train_dataset. "
                "Either provide input_features/output_features during __init__, "
                "or ensure a DataModule is attached to the trainer."
            )
            raise RuntimeError(msg)

        # Import here to avoid circular dependency
        from lerobot.datasets.utils import dataset_to_policy_features

        # Get the training dataset
        train_dataset = self.trainer.datamodule.train_dataset

        # Extract features from LeRobot dataset
        if not hasattr(train_dataset, "_lerobot_dataset"):
            msg = "DataModule's train_dataset must have '_lerobot_dataset' attribute. Are you using LeRobotDataModule?"
            raise RuntimeError(msg)

        lerobot_dataset = train_dataset._lerobot_dataset

        # Convert LeRobot dataset features to policy features
        features = dataset_to_policy_features(lerobot_dataset.meta.features)

        # Get dataset statistics if not provided
        stats = self._dataset_stats
        if stats is None:
            stats = lerobot_dataset.meta.stats

        # Initialize policy now
        self._initialize_policy(features, features, self._provided_config, stats)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the LeRobot policy.

        Args:
            batch: Input batch containing observations and actions.

        Returns:
            Policy output (format depends on policy type).
        """
        if self.lerobot_policy is None:
            msg = "Policy not initialized. Call setup() or provide input_features during __init__."
            raise RuntimeError(msg)
        return self.lerobot_policy.forward(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for Lightning.

        Args:
            batch: Training batch.
            batch_idx: Batch index.

        Returns:
            Scalar loss value.
        """
        output = self.lerobot_policy.forward(batch)

        # Handle different output formats from LeRobot policies
        if isinstance(output, tuple) and len(output) == 2:
            _, loss_dict = output
        elif isinstance(output, dict):
            loss_dict = output
        else:
            # Some policies might return just a loss tensor
            return output

        # Sum all loss components
        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

        # Log individual loss components
        if isinstance(loss_dict, dict):
            for key, val in loss_dict.items():
                self.log(f"train/{key}", val, prog_bar=True)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for Lightning.

        Args:
            batch: Validation batch.
            batch_idx: Batch index.

        Returns:
            Scalar loss value.
        """
        output = self.lerobot_policy.forward(batch)

        # Handle different output formats
        if isinstance(output, tuple) and len(output) == 2:
            _, loss_dict = output
        elif isinstance(output, dict):
            loss_dict = output
        else:
            return output

        # Sum all loss components
        loss = sum(loss_dict.values()) if isinstance(loss_dict, dict) else loss_dict

        # Log individual loss components
        if isinstance(loss_dict, dict):
            for key, val in loss_dict.items():
                self.log(f"val/{key}", val, prog_bar=True)

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Args:
            batch: Input batch with observations.

        Returns:
            Predicted actions.
        """
        return self.lerobot_policy.select_action(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for Lightning.

        Uses LeRobot's parameter grouping if available (e.g., for backbone learning rates).

        Returns:
            Configured optimizer.
        """
        # Check if the policy has custom parameter grouping
        if hasattr(self.lerobot_policy, "get_optim_params"):
            param_groups = self.lerobot_policy.get_optim_params()
            # If get_optim_params returns a list of dicts, use it directly
            # Otherwise, wrap in a list
            if isinstance(param_groups, list) and param_groups and isinstance(param_groups[0], dict):
                return torch.optim.Adam(param_groups, lr=self.learning_rate)
            else:
                return torch.optim.Adam(param_groups, lr=self.learning_rate)
        else:
            # Default: optimize all parameters
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @property
    def config(self) -> PreTrainedConfig:
        """Access the underlying LeRobot config.

        Returns:
            The policy's configuration object.
        """
        return self._config

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  policy_name={self.policy_name!r},\n"
            f"  policy_class={self.lerobot_policy.__class__.__name__},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  stats={'provided' if self.stats is not None else 'None'}\n"
            f")"
        )
