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
from lightning_utilities import module_available

from getiaction.data import Observation
from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.policies.base import Policy
from getiaction.policies.lerobot.mixin import LeRobotFromConfig

if TYPE_CHECKING:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pretrained import PreTrainedPolicy

    from getiaction.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    LEROBOT_AVAILABLE = True
else:
    PreTrainedConfig = None
    FeatureType = None
    LeRobotDataset = None
    LeRobotDatasetMetadata = None
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

    Supported policies (from LeRobot):
        - act: Action Chunking Transformer for imitation learning
        - diffusion: Diffusion Policy for behavior cloning
        - vqbet: VQ-BeT (Vector Quantized Behavior Transformer)
        - tdmpc: TD-MPC (Temporal Difference Model Predictive Control)
        - sac: Soft Actor-Critic for reinforcement learning
        - pi0: PI0 Vision-Language-Action policy
        - pi05: PI0.5 (improved PI0 with better performance)
        - smolvla: SmolVLA (Small Vision-Language-Action model)
        - groot: GR00T policy for generalist robots
        - xvla: XVLA (Cross-modal Vision-Language-Action)
        - reward_classifier: Reward classifier for RL

    Example:
        >>> # Option 1: Load pretrained from HuggingFace Hub
        >>> policy = LeRobotPolicy.from_pretrained("lerobot/act_pusht")

        >>> # Option 2: Eager initialization from dataset/repo
        >>> policy = LeRobotPolicy.from_dataset("act", "lerobot/pusht", optimizer_lr=1e-4)

        >>> # Option 3: Lazy initialization (features extracted in setup())
        >>> policy = LeRobotPolicy(policy_name="diffusion", num_inference_steps=100)

        >>> # Option 4: Eager with explicit features
        >>> policy = LeRobotPolicy(
        ...     policy_name="act",
        ...     input_features=input_features,
        ...     output_features=output_features,
        ...     dataset_stats=dataset.meta.stats,
        ...     optimizer_lr=1e-4,  # Any LeRobot config param can be passed
        ... )

        >>> # Option 5: Use with LightningCLI (lazy init)
        >>> # model:
        >>> #   class_path: getiaction.policies.lerobot.LeRobotPolicy
        >>> #   init_args:
        >>> #     policy_name: act
        >>> #     policy_config:
        >>> #       optimizer_lr: 0.0001

    Args:
        policy_name: Name of the LeRobot policy to instantiate (e.g., 'act', 'diffusion').
        input_features: Dictionary of input feature definitions (PolicyFeature objects).
        output_features: Dictionary of output feature definitions (PolicyFeature objects).
        config: Pre-built LeRobot config object. If provided, policy_config is ignored.
        dataset_stats: Dataset statistics for normalization.
        policy_config: Policy-specific parameters as a dict (for YAML configs).
        **kwargs: Policy-specific parameters (for Python usage). Merged with policy_config;
            kwargs take precedence if both specify the same key.

    Raises:
        ImportError: If LeRobot is not installed.
        ValueError: If policy_name is not recognized by LeRobot.

    Note:
        This is the "universal wrapper" approach. For explicit parameter definitions
        and better IDE support, consider using specific wrappers (e.g., ACT).
    """

    def __init__(
        self,
        policy_name: str,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        config: PreTrainedConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        policy_config: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
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
            policy_config: Policy-specific parameters as a dict (for YAML configs).
            **kwargs: Policy-specific parameters (for Python usage). Merged with
                policy_config; kwargs take precedence if both specify the same key.

        Raises:
            ImportError: If LeRobot is not installed.
            ValueError: If policy_name is not recognized by LeRobot.
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

        super().__init__()

        # Store metadata
        self.policy_name = policy_name

        # Merge policy_config (from YAML) with kwargs (from Python)
        # kwargs take precedence for CLI override support
        merged_policy_config = {**(policy_config or {}), **kwargs}

        # Store for lazy initialization
        self._input_features = input_features
        self._output_features = output_features
        self._provided_config = config
        self._dataset_stats = dataset_stats
        self._policy_config = merged_policy_config

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

    @classmethod
    def from_dataset(
        cls,
        policy_name: str,
        dataset: LeRobotDataset | str,
        **kwargs: Any,  # noqa: ANN401
    ) -> "LeRobotPolicy":
        """Create policy with eager initialization from a dataset or repo ID.

        This factory method extracts features from the dataset and builds the policy
        immediately, making it ready for inference without a Lightning Trainer.

        Args:
            policy_name: Name of the policy ('act', 'diffusion', 'vqbet', etc.)
            dataset: Either a LeRobotDataset instance or a HuggingFace Hub repo ID string.
                If a string is provided, only metadata is fetched (lightweight).
            **kwargs: Additional policy configuration parameters.

        Returns:
            Fully initialized LeRobotPolicy ready for inference.

        Examples:
            Create from repo ID (lightweight, only fetches metadata):

                >>> policy = LeRobotPolicy.from_dataset(
                ...     "act",
                ...     "lerobot/pusht",
                ...     dim_model=512,
                ... )

            Create from an already loaded dataset:

                >>> dataset = LeRobotDataset("lerobot/pusht")
                >>> policy = LeRobotPolicy.from_dataset(
                ...     "diffusion",
                ...     dataset,
                ...     num_inference_steps=100,
                ... )
        """
        if not LEROBOT_AVAILABLE:
            msg = "LeRobotPolicy.from_dataset requires LeRobot to be installed."
            raise ImportError(msg)

        # Get metadata from dataset or repo ID
        if isinstance(dataset, str):
            meta = LeRobotDatasetMetadata(dataset)
        else:
            meta = dataset.meta

        # Convert dataset features to policy features
        features = dataset_to_policy_features(meta.features)

        # Split into input/output features
        input_features = {k: f for k, f in features.items() if f.type != FeatureType.ACTION}
        output_features = {k: f for k, f in features.items() if f.type == FeatureType.ACTION}

        return cls(
            policy_name=policy_name,
            input_features=input_features,
            output_features=output_features,
            dataset_stats=meta.stats,
            **kwargs,
        )

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

            # Remove dataset_stats from policy_config if present
            # (it should be passed to policy constructor, not config)
            clean_policy_config = {k: v for k, v in self._policy_config.items() if k != "dataset_stats"}

            # Create config dynamically using LeRobot's factory
            config = make_policy_config(
                self.policy_name,
                input_features=input_features,
                output_features=output_features,
                **clean_policy_config,
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for Lightning.

        Uses LeRobot's get_optimizer_preset from config which includes proper
        parameter grouping (e.g., different lr for backbone) and optimizer settings.

        Returns:
            Configured optimizer from LeRobot's preset.
        """
        # Use LeRobot's optimizer preset from config - this handles:
        # - Proper optimizer type (Adam, AdamW, etc.)
        # - Learning rate and weight decay from config
        # - Parameter grouping via get_optim_params (e.g., backbone lr)
        optimizer_config = self._config.get_optimizer_preset()
        params = self.lerobot_policy.get_optim_params()
        return optimizer_config.build(params)

    def training_step(self, batch: Observation | dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: Input batch (Observation or LeRobot dict format).
            batch_idx: Index of the batch.

        Returns:
            The total loss for the batch.
        """
        del batch_idx  # Unused argument

        # LeRobot policies return (loss, loss_dict) where loss_dict may be None (e.g., Diffusion)
        total_loss, loss_dict = self.lerobot_policy(batch)

        # Log individual loss components if available
        if loss_dict is not None:
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

    def forward(  # type: ignore[override]
        self,
        batch: Observation | dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward pass for universal LeRobot policy.

        Handles both training and inference modes:
        - Training: Returns (loss, loss_dict) for backpropagation
        - Inference: Returns denormalized actions for prediction/export

        Args:
            batch: Input batch (Observation or LeRobot dict).

        Returns:
            - Training mode: Tuple of (loss, loss_dict or None)
            - Inference mode: Action tensor
        """
        if self.training:
            # Training mode: Use LeRobot policy's forward() which computes loss
            # Convert to LeRobot format if needed
            batch_dict = FormatConverter.to_lerobot_dict(batch) if isinstance(batch, Observation) else batch

            # Call underlying policy forward
            output = self.lerobot_policy(batch_dict)

            # Handle different return formats (some policies return tuple, some just loss)
            if isinstance(output, tuple):
                return output
            return output, None

        # Inference mode: Generate actions via select_action
        return self.select_action(batch)

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
        if policy_name is None and hasattr(self, "_lerobot_policy"):
            policy_name = self._lerobot_policy.config.type

        # Get learning rate from config if available
        lr_info = "N/A"
        if hasattr(self, "_config") and self._config is not None:
            optimizer_preset = self._config.get_optimizer_preset()
            lr_info = f"{optimizer_preset.lr}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  policy_name={policy_name!r},\n"
            f"  lr={lr_info},\n"
            f")"
        )
