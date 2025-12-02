# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SmolVLAAction Chunking Transformer (ACT) policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's ACT implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lerobot.configs.types import NormalizationMode
from lightning_utilities.core.imports import module_available

from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.policies.base import Policy
from getiaction.policies.lerobot.mixin import LeRobotFromConfig
from getiaction.policies.lerobot.smolvla_with_xai import SmolVLAPolicyWithXAI

if TYPE_CHECKING:
    from torch import nn

    from getiaction.data import Observation
    from getiaction.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as _LeRobotSmolVLAConfig

    LEROBOT_AVAILABLE = True
else:
    LeRobotDataset = None
    dataset_to_policy_features = None
    _LeRobotACTConfig = None
    _LeRobotACTPolicy = None
    LEROBOT_AVAILABLE = False


class SmolVLA(Policy, LeRobotFromConfig):
    """LeRobot's SmolVLA policy wrapper with explainability.

    PyTorch Lightning wrapper around LeRobot's SmolVLA implementation that provides
    flexible configuration options and seamless integration with the getiaction
    data pipeline and explainability. The policy supports lazy initialization, where the LeRobot
    policy is created during setup() after dataset features are loaded.

    SmolVLA is a lightweight open-source VLA policy for robotics:
    given multi-camera RGB input + current robot state + a natural-language instruction,
    the model outputs a chunk of continuous control commands.
    Designed for efficient deployment on modest hardware.

    The wrapper supports multiple configuration methods through the ``LeRobotFromConfig`` mixin.
    See ``LeRobotFromConfig`` for detailed configuration examples.

    Examples:
        Basic usage with explicit arguments (recommended):
            >>> from getiaction.policies.lerobot import SmolVLA
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> from getiaction.train import Trainer

            >>> # Create policy with explicit parameters
            >>> policy = SmolVLA()

            >>> # Create datamodule
            >>> datamodule = LeRobotDataModule(
            ...     repo_id="lerobot/pusht",
            ...     train_batch_size=8,
            ... )

            >>> # Train
            >>> trainer = Trainer(max_epochs=100)
            >>> trainer.fit(policy, datamodule)

        Using configuration file (alternative):
            >>> # From dict, YAML, Pydantic, or LeRobot config
            >>> policy = SmolVLA.from_config("config.yaml")
            >>> # or
            >>> policy = SmolVLA.from_config({"dim_model": 512, "chunk_size": 100})

        Using pre-configured LeRobot config (advanced):
            >>> from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as LeRobotSmolVLAConfig
            >>> lerobot_config = LeRobotSmolVLAConfig()
            >>> policy = SmolVLA.from_config(lerobot_config)

        YAML configuration with LightningCLI (explicit args):

            ```yaml
            # config.yaml
            model:
              class_path: getiaction.policies.lerobot.SmolVLA
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
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Input / output structure.
        n_obs_steps: int = 1,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        normalization_mapping: dict[str, NormalizationMode] | None = None,

        # State dimensions
        max_state_dim: int = 32,
        max_action_dim: int = 32,

        # Image preprocessing
        resize_imgs_with_padding: tuple[int, int] = (512, 512),

        # Aloha specific parameters
        empty_cameras: int = 0,
        adapt_to_pi_aloha: bool = False,
        use_delta_joint_actions_aloha: bool = False,

        # Tokenizer and decoding
        tokenizer_max_length: int = 48,
        num_steps: int = 10,
        use_cache: bool = True,

        # Finetuning settings
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        train_state_proj: bool = True,

        # Training presets
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-10,
        optimizer_grad_clip_norm: float = 10,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int = 30_000,
        scheduler_decay_lr: float = 2.5e-6,

        # VLM/VLA model parameters
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = False,
        add_image_special_tokens: bool = False,
        attention_mode: str = "cross_attn",
        prefix_length: int = -1,
        pad_language_to: str = "longest",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,

        # Positional encoding
        min_period: float = 4e-3,
        max_period: float = 4.0,

        # XAI
        layer_idx: int | None = -1,
        head_idx: int | None = None,

        # Additional parameters via kwargs
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize SmolVLAWithXAI policy wrapper.

        The policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        For loading from dicts, YAML, Pydantic models, or LeRobot configs, use the
        inherited methods from LeRobotFromConfig mixin: ``from_config()``,
        ``from_dict()``, ``from_yaml()``, ``from_pydantic()``, ``from_dataclass()``,
        or ``from_lerobot_config()``.

        Args:
            n_obs_steps: Number of observation steps.
            chunk_size: Size of the output chunk.
            n_action_steps: Number of action to predict.
            normalization_mapping: Choosing the normalization mode.
            max_state_dim: State token dimensionality. Shorter state and action vectors will be padded.
            max_action_dim: Action token dimensionality.
            resize_imgs_with_padding: Default input image size
            empty_cameras: Add empty images. Used by smolvla_aloha_sim which adds the empty left and right wrist cameras
                in addition to the top camera.
            adapt_to_pi_aloha: Converts the joint and gripper values from the standard Aloha space to the space used by
                the pi internal runtime which was used to train the base model.,
            use_delta_joint_actions_aloha: Converts joint dimensions to deltas with respect to the current state before
                passing to the model. Gripper dimensions will remain in absolute values.
            tokenizer_max_length: Tokenizer.
            num_steps: Decoding.
            use_cache: Attention utils.
            freeze_vision_encoder: Freeze the encoder.
            train_expert_only: Train only the expert part of the model.
            train_state_proj: Train state projection.
            optimizer_lr: Learning rate for the internal optimizer.
            optimizer_betas: Betas for the internal optimizer.
            optimizer_eps: Epsilon for the internal optimizer.
            optimizer_weight_decay: Weight decay for the internal optimizer.
            optimizer_grad_clip_norm: Gradient clipping for the internal optimizer.
            scheduler_warmup_steps: Warmup steps for the internal scheduler.
            scheduler_decay_steps: Decay steps for the internal scheduler.
            scheduler_decay_lr: Learning rate decay for the internal scheduler.
            vlm_model_name: Select the VLM backbone.
            load_vlm_weights: Set to True in case of training the expert from scratch. True when init from pretrained
                SmolVLA weights.
            add_image_special_tokens: Whether to use special image tokens around image features.
            attention_mode: Attention mode.
            prefix_length: Prefix length.
            pad_language_to: Padding.
            num_expert_layers: Less or equal to 0 is the default where the action expert has the same number of layers
                of VLM. Otherwise, the expert has fewer layers.
            num_vlm_layers: Number of layers used in the VLM (first num_vlm_layers layers)
            self_attn_every_n_layers: Interleave SA layers each self_attn_every_n_layers.
            expert_width_multiplier: The action expert hidden size (wrt to the VLM)
            min_period: Sensitivity min for the timestep used in sine-cosine positional encoding.
            max_period: Sensitivity max for the timestep used in sine-cosine positional encoding.
            layer_idx: Specifies which layer to use for XAI (set to none for average)
            head_idx: Specifies which attention head to use for XAI (set to none for average or -1 for max)
            **kwargs: Additional SmolVLA parameters .

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "SmolVLA requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__()

        # Create default value for normalization_mapping if it was not provided
        if normalization_mapping is None:
            normalization_mapping: dict[str, NormalizationMode] = {
                    "VISUAL": NormalizationMode.IDENTITY,
                    "STATE": NormalizationMode.MEAN_STD,
                    "ACTION": NormalizationMode.MEAN_STD,
            }

        # Build config dict from explicit args
        self._config_object = None
        self._config_kwargs = {
            "n_obs_steps": n_obs_steps,
            "chunk_size": chunk_size,
            "n_action_steps": n_action_steps,
            "normalization_mapping": normalization_mapping,
            "max_state_dim": max_state_dim,
            "max_action_dim": max_action_dim,
            "resize_imgs_with_padding": resize_imgs_with_padding,
            "empty_cameras": empty_cameras,
            "adapt_to_pi_aloha": adapt_to_pi_aloha,
            "use_delta_joint_actions_aloha": use_delta_joint_actions_aloha,
            "tokenizer_max_length": tokenizer_max_length,
            "num_steps": num_steps,
            "use_cache": use_cache,
            "freeze_vision_encoder": freeze_vision_encoder,
            "train_expert_only": train_expert_only,
            "train_state_proj": train_state_proj,
            "optimizer_lr": optimizer_lr,
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "optimizer_grad_clip_norm": optimizer_grad_clip_norm,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            "scheduler_decay_steps": scheduler_decay_steps,
            "scheduler_decay_lr": scheduler_decay_lr,
            "vlm_model_name": vlm_model_name,
            "load_vlm_weights": load_vlm_weights,
            "add_image_special_tokens": add_image_special_tokens,
            "attention_mode": attention_mode,
            "prefix_length": prefix_length,
            "pad_language_to": pad_language_to,
            "num_expert_layers": num_expert_layers,
            "num_vlm_layers": num_vlm_layers,
            "self_attn_every_n_layers": self_attn_every_n_layers,
            "expert_width_multiplier": expert_width_multiplier,
            "min_period": min_period,
            "max_period": max_period,
            ** kwargs,
        }

        self.learning_rate = optimizer_lr
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self._framework = "lerobot"

        # Policy will be initialized in setup()
        self._smolvla_policy_with_xai: SmolVLAPolicyWithXAI
        self.model: nn.Module | None = None

        self.save_hyperparameters()

    @property
    def smolvla_policy_with_xai(self) -> SmolVLAPolicyWithXAI:
        """Get the initialized wrapped policy.

        Returns:
            The initialized SmolVLAPolicyWithXAI

        Raises:
            RuntimeError: If the policy hasn't been initialized yet.
        """
        if not hasattr(self, "_smolvla_policy_with_xai") or self._smolvla_policy_with_xai is None:
            msg = "Policy not initialized. Call setup() first."
            raise RuntimeError(msg)
        return self._smolvla_policy_with_xai

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

        if hasattr(self, "_smolvla_policy_with_xai") and self.smolvla_policy_with_xai is not None:
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

        # Create or update LeRobot SmolVLA configuration based on what user provided
        if self._config_object is not None:
            # User provided a full config object - update input/output features
            lerobot_config = self._config_object
            lerobot_config.input_features = features
            lerobot_config.output_features = features
        else:
            # User provided dict or explicit args - create config
            lerobot_config = _LeRobotSmolVLAConfig(  # type: ignore[misc]
                input_features=features,
                output_features=features,
                **self._config_kwargs,  # type: ignore[arg-type]
            )

        # Initialize the policy
        policy = SmolVLAPolicyWithXAI(lerobot_config,
                                      dataset_stats=stats,
                                      layer_idx=self.layer_idx,
                                      head_idx=self.head_idx)  # type: ignore[arg-type,misc]
        self.add_module("_smolvla_policy_with_xai", policy)
        self.model = policy.model

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for the SmolVLA policy.

        The return value depends on the model's training mode:
        - In training mode: Returns (loss, loss_dict) from LeRobot's forward method
        - In evaluation mode: Returns action predictions via select_action method

        Args:
            batch (Observation): Input batch of observations

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]: In training mode,
                returns tuple of (loss, loss_dict). In eval mode, returns selected actions tensor.
        """
        # Convert to LeRobot format for internal processing
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        if self.training:
            # During training, return loss information for backpropagation
            return self.smolvla_policy_with_xai(batch_dict)

        # During evaluation, return action predictions
        return self.select_action(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's custom parameter groups.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        return torch.optim.AdamW(self.smolvla_policy_with_xai.get_optim_params(), lr=self.learning_rate)

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

        total_loss, loss_dict = self.smolvla_policy_with_xai(batch_dict)
        for key, value in loss_dict.items():
            if key == "loss":
                self.log(f"train/{key}", value, prog_bar=True)
            else:
                self.log(f"train/{key}", value.mean().item(), prog_bar=False)
        return total_loss

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step of the policy.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step of the policy.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout evaluation.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Converts the Observation to LeRobot dict format and passes it to the
        underlying LeRobot policy for action prediction.

        Args:
            batch: Input batch of observations.

        Returns:
            The selected action tensor.
        """
        # Move batch to device (observations from gym are on CPU)
        batch = batch.to(self.device)

        # Convert to LeRobot format
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        return self.smolvla_policy_with_xai.select_action(batch_dict)

    def reset(self) -> None:
        """Reset the policy state."""
        self.smolvla_policy_with_xai.reset()
