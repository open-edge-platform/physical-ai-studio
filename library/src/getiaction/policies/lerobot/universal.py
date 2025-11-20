# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Universal wrapper for any LeRobot policy.

This module provides a generic wrapper that can instantiate any LeRobot policy
dynamically without requiring explicit wrappers for each policy type.

For users who prefer explicit wrappers with full parameter definitions and IDE
support, see the specific policy modules (e.g., ACT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from lightning_utilities import module_available

from getiaction.data import Observation
from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.policies.base import Policy
from getiaction.policies.lerobot.mixin import LeRobotFromConfig

if TYPE_CHECKING:
    from lerobot.configs.types import PolicyFeature
    from lerobot.policies.pretrained import PreTrainedPolicy

    from getiaction.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    LEROBOT_AVAILABLE = True
else:
    PreTrainedConfig = None
    dataset_to_policy_features = None
    get_policy_class = None
    make_policy_config = None
    make_pre_post_processors = None
    LEROBOT_AVAILABLE = False


class LeRobotPolicy(Policy, LeRobotFromConfig):
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
        >>> # Option 1: Load pretrained model
        >>> policy = LeRobotPolicy.from_pretrained(
        ...     "lerobot/diffusion_pusht"
        ... )

        >>> # Option 2: Pass config as dict
        >>> policy = LeRobotPolicy(
        ...     policy_name="diffusion",
        ...     input_features=features,
        ...     output_features=features,
        ...     num_steps=100,
        ...     noise_scheduler="ddpm",
        ...     stats=dataset.meta.stats,
        ... )

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

    SUPPORTED_POLICIES: ClassVar[list[str]] = [
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
        **extra_config_kwargs: Any,  # noqa: ANN401
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

        Raises:
            ImportError: If LeRobot is not installed.
            ValueError: If policy_name is not supported.

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
        self._lerobot_policy: PreTrainedPolicy
        self._config: PreTrainedConfig | None = None

        # If features are provided, initialize immediately (backward compatibility)
        if input_features is not None and output_features is not None:
            self._initialize_policy(input_features, output_features, config, dataset_stats)
        elif config is not None:
            # Config provided directly - can initialize now
            self._initialize_policy(None, None, config, dataset_stats)

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> PreTrainedPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot policy.

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if not hasattr(self, "_lerobot_policy") or self._lerobot_policy is None:
            msg = "Policy not initialized. Call setup() or provide input_features during __init__."
            raise RuntimeError(msg)
        return self._lerobot_policy

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

        Raises:
            ValueError: If neither config nor features are provided.
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
        policy = policy_cls(config)
        self.add_module("_lerobot_policy", policy)

        # Use LeRobot policy directly as model (it's already an nn.Module)
        self.model = self._lerobot_policy

        # Create preprocessor/postprocessor for normalization
        self._preprocessor, self._postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)

        # Expose framework info
        self._framework = "lerobot"
        self._config = config

    def setup(self, stage: str) -> None:
        """Lightning hook called before training/validation/test.

        Extracts input/output features from the DataModule's dataset if not provided
        during initialization. This enables YAML-based configuration without requiring
        features to be specified in advance.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').

        Raises:
            RuntimeError: If DataModule or train_dataset is not available.
        """
        del stage  # Unused argument

        if hasattr(self, "_lerobot_policy") and self._lerobot_policy is not None:
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

        # Get the training dataset - handle both data formats
        train_dataset = self.trainer.datamodule.train_dataset

        # Extract LeRobot dataset based on type
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            # Wrapped in adapter for getiaction format conversion
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif hasattr(train_dataset, "meta") and hasattr(train_dataset.meta, "features"):
            # Assume it's a raw LeRobotDataset (data_format="lerobot")
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule with appropriate data_format."
            )
            raise RuntimeError(msg)

        # Convert LeRobot dataset features to policy features
        features = dataset_to_policy_features(lerobot_dataset.meta.features)

        # Get dataset statistics if not provided
        stats = self._dataset_stats
        if stats is None:
            stats = lerobot_dataset.meta.stats

        # Initialize policy now
        self._initialize_policy(features, features, self._provided_config, stats)

    def _process_loss_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor, dict | None] | dict,
        log_prefix: str,
    ) -> torch.Tensor:
        """Process and log loss output from LeRobot policy.

        Handles the different output formats from LeRobot policies and logs losses appropriately.

        Args:
            output: Policy output, can be one of:
                1. Tuple[Tensor, dict]: Standard format used by most policies (ACT, VQ-BeT, TDMPC, etc.)
                   Returns (loss, loss_dict) where loss_dict contains individual loss components.
                   Example: ACT returns (loss, {"l1_loss": ..., "kld_loss": ...})

                2. Tuple[Tensor, None]: Used by policies without detailed loss breakdown (Diffusion).
                   Returns (loss, None) where None indicates no additional loss components.

                3. dict: Some policies may return only a dictionary of losses.
                   The dictionary contains loss components that need to be summed.

                4. Tensor: Direct loss tensor (legacy/simple policies).
                   Returns just the scalar loss tensor.

            log_prefix: Prefix for logging (e.g., "train" or "val")

        Returns:
            Total loss tensor

        Raises:
            ValueError: If loss dictionary is empty

        Note:
            See module docstring for detailed documentation of each output format.
        """
        if isinstance(output, tuple) and len(output) == 2:  # noqa: PLR2004
            loss, loss_dict = output
            if loss_dict is None:
                # Case 2: Diffusion-style output with no loss breakdown
                self.log(f"{log_prefix}/loss", loss, prog_bar=True)
                return loss
            # Case 1: Standard output with loss dict - continue processing below
        elif isinstance(output, dict):
            # Case 3: Dictionary of losses
            loss_dict = output
        else:
            # Case 4: Direct loss tensor
            return output

        # Sum all loss components from loss_dict
        if isinstance(loss_dict, dict):
            if not loss_dict:
                msg = "Loss dictionary is empty - policy returned no loss components"
                raise ValueError(msg)
            # Use torch.stack + sum to maintain tensor type (avoids int return from sum())
            loss = torch.stack(list(loss_dict.values())).sum()
            # Log individual loss components
            for key, val in loss_dict.items():
                self.log(f"{log_prefix}/{key}", val, prog_bar=True)
        else:
            loss = loss_dict

        self.log(f"{log_prefix}/loss", loss, prog_bar=True)
        return loss

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
            return torch.optim.Adam(param_groups, lr=self.learning_rate)
        # Default: optimize all parameters
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Observation | dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch (Observation | dict[str, torch.Tensor]): Input batch of observations.
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        total_loss, loss_dict = self.lerobot_policy(batch)
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, prog_bar=False)
        self.log("train/loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step for Lightning.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step for Lightning.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Batch index.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def select_action(self, batch: Observation | dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Converts the Observation to LeRobot dict format, applies preprocessing,
        gets action prediction, and applies postprocessing (denormalization).

        Args:
            batch: Input batch of observations (raw, from gym).

        Returns:
            Predicted actions (denormalized).
        """
        # Convert to LeRobot format if needed
        batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch

        # Apply preprocessing
        batch_dict = self._preprocessor(batch_dict)

        # Get action from policy
        action = self.lerobot_policy.select_action(batch_dict)

        # Apply postprocessing
        return self._postprocessor(action)

    def reset(self) -> None:
        """Reset the policy state for a new episode.

        Forwards the reset call to the underlying LeRobot policy,
        which clears action queues, observation histories, and any
        other stateful components.
        """
        self.lerobot_policy.reset()

    @property
    def config(self) -> PreTrainedConfig:
        """Access the underlying LeRobot config.

        Returns:
            The policy's configuration object.
        """
        return self._config

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String summarizing the policy instance.
        """
        # Get policy_name from either attribute or config
        policy_name = getattr(self, "policy_name", None)
        if policy_name is None and hasattr(self, "lerobot_policy"):
            policy_name = self.lerobot_policy.config.type

        # Get stats info
        stats_info = "None"
        if hasattr(self, "lerobot_policy") and hasattr(self.lerobot_policy, "dataset_stats"):
            stats_info = "provided" if self.lerobot_policy.dataset_stats is not None else "None"

        return (
            f"{self.__class__.__name__}(\n"
            f"  policy_name={policy_name!r},\n"
            f"  policy_class={self.lerobot_policy.__class__.__name__},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  stats={stats_info}\n"
            f")"
        )
