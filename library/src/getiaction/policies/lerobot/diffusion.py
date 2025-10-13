# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's Diffusion Policy implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning_utilities.core.imports import module_available

from getiaction.data.lerobot import FormatConverter
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
from getiaction.data.observation import GymObservation
from getiaction.policies.base import Policy
from getiaction.policies.lerobot.mixin import LeRobotFromConfig

if TYPE_CHECKING:
    from torch import nn

    from getiaction.data import Observation

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as _LeRobotDiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as _LeRobotDiffusionPolicy

    LEROBOT_AVAILABLE: bool = True
else:
    LEROBOT_AVAILABLE = False
    _LeRobotDiffusionPolicy = None
    _LeRobotDiffusionConfig = None
    LeRobotDataset = None
    dataset_to_policy_features = None


class Diffusion(Policy, LeRobotFromConfig):
    """Diffusion Policy from LeRobot with lazy initialization.

    This class wraps LeRobot's Diffusion Policy implementation, providing a
    Lightning-compatible interface. The underlying LeRobot policy is created
    lazily in setup() after the dataset is loaded, enabling both YAML-based
    configuration and direct Python API usage.

    The Diffusion Policy uses denoising diffusion probabilistic models to
    generate robot actions through iterative denoising of Gaussian noise.

    Attributes:
        lerobot_policy: The underlying LeRobot DiffusionPolicy instance.
            Created during setup() and used for forward passes.

    Example:
        Basic usage with Python API:

        >>> from getiaction.policies.lerobot import Diffusion
        >>> from getiaction.data.lerobot import LeRobotDataModule
        >>> import lightning as L

        >>> # Create policy with default parameters
        >>> policy = Diffusion(
        ...     n_obs_steps=2,
        ...     horizon=16,
        ...     n_action_steps=8,
        ... )

        >>> # Create data module
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/pusht",
        ...     batch_size=64,
        ... )

        >>> # Train with Lightning
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(policy, datamodule)

        Advanced configuration with custom architecture:

        >>> policy = Diffusion(
        ...     n_obs_steps=2,
        ...     horizon=16,
        ...     n_action_steps=8,
        ...     down_dims=(256, 512, 1024),
        ...     vision_backbone="resnet34",
        ...     num_train_timesteps=100,
        ...     learning_rate=1e-4,
        ... )

        Using YAML configuration:

        ```yaml
        model:
          class_path: getiaction.policies.lerobot.Diffusion
          init_args:
            n_obs_steps: 2
            horizon: 16
            n_action_steps: 8
            down_dims: [512, 1024, 2048]
            vision_backbone: resnet18
            learning_rate: 1e-4
        data:
          class_path: getiaction.data.lerobot.LeRobotDataModule
          init_args:
            repo_id: lerobot/pusht
            batch_size: 64
        ```

        Then run from command line:

        ```bash
        getiaction fit --config config.yaml
        ```

    Note:
        This class requires LeRobot to be installed. Install with:
        ``pip install lerobot`` or ``pip install getiaction[lerobot]``
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        # Input/output structure
        n_obs_steps: int = 2,
        horizon: int = 16,
        n_action_steps: int = 8,
        drop_n_last_frames: int = 7,
        # Vision backbone
        vision_backbone: str = "resnet18",
        crop_shape: tuple[int, int] | None = (84, 84),
        crop_is_random: bool = True,
        pretrained_backbone_weights: str | None = None,
        use_group_norm: bool = True,
        spatial_softmax_num_keypoints: int = 32,
        use_separate_rgb_encoder_per_camera: bool = False,
        # U-Net architecture
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        use_film_scale_modulation: bool = True,
        # Noise scheduler
        noise_scheduler_type: str = "DDPM",
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        # Inference
        num_inference_steps: int | None = None,
        # Loss computation
        do_mask_loss_for_padding: bool = False,
        # Optimizer
        learning_rate: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.95, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-6,
        # Scheduler
        scheduler_name: str = "cosine",
        scheduler_warmup_steps: int = 500,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Diffusion policy wrapper.

        The LeRobot policy is created lazily in setup() after the dataset is loaded.
        This is called automatically by Lightning before training begins.

        Args:
            n_obs_steps: Number of environment steps worth of observations to pass to the policy.
            horizon: Diffusion model action prediction size.
            n_action_steps: Number of action steps to execute per forward pass.
            drop_n_last_frames: Frames to drop from the end to avoid excessive padding.
            vision_backbone: Vision encoder architecture (e.g., "resnet18").
            crop_shape: (H, W) shape to crop images to. None means no cropping.
            crop_is_random: Whether to use random crop during training.
            pretrained_backbone_weights: Pretrained weights for vision backbone.
            use_group_norm: Whether to use group normalization in the backbone.
            spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
            use_separate_rgb_encoder_per_camera: Whether to use separate encoders per camera.
            down_dims: Feature dimensions for each U-Net downsampling stage.
            kernel_size: Convolutional kernel size in U-Net.
            n_groups: Number of groups for group normalization in U-Net.
            diffusion_step_embed_dim: Embedding dimension for diffusion timestep.
            use_film_scale_modulation: Whether to use FiLM scale modulation.
            noise_scheduler_type: Type of noise scheduler ("DDPM" or "DDIM").
            num_train_timesteps: Number of diffusion steps for forward diffusion.
            beta_schedule: Name of the beta schedule.
            beta_start: Beta value for the first diffusion step.
            beta_end: Beta value for the last diffusion step.
            prediction_type: Type of prediction ("epsilon" or "sample").
            clip_sample: Whether to clip samples during inference.
            clip_sample_range: Magnitude of the clipping range.
            num_inference_steps: Number of reverse diffusion steps at inference time.
            do_mask_loss_for_padding: Whether to mask loss for padded actions.
            learning_rate: Learning rate for optimizer.
            optimizer_betas: Beta parameters for Adam optimizer.
            optimizer_eps: Epsilon for Adam optimizer.
            optimizer_weight_decay: Weight decay for optimizer.
            scheduler_name: Name of learning rate scheduler.
            scheduler_warmup_steps: Number of warmup steps for scheduler.
            **kwargs: Additional arguments passed to DiffusionConfig.

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "Diffusion requires LeRobot framework.\n\nInstall with:\n    uv pip install lerobot\n"
            raise ImportError(msg)

        super().__init__()

        # Store configuration for use in setup()
        self._config_kwargs = {
            "n_obs_steps": n_obs_steps,
            "horizon": horizon,
            "n_action_steps": n_action_steps,
            "drop_n_last_frames": drop_n_last_frames,
            "vision_backbone": vision_backbone,
            "crop_shape": crop_shape,
            "crop_is_random": crop_is_random,
            "pretrained_backbone_weights": pretrained_backbone_weights,
            "use_group_norm": use_group_norm,
            "spatial_softmax_num_keypoints": spatial_softmax_num_keypoints,
            "use_separate_rgb_encoder_per_camera": use_separate_rgb_encoder_per_camera,
            "down_dims": down_dims,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "diffusion_step_embed_dim": diffusion_step_embed_dim,
            "use_film_scale_modulation": use_film_scale_modulation,
            "noise_scheduler_type": noise_scheduler_type,
            "num_train_timesteps": num_train_timesteps,
            "beta_schedule": beta_schedule,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "prediction_type": prediction_type,
            "clip_sample": clip_sample,
            "clip_sample_range": clip_sample_range,
            "num_inference_steps": num_inference_steps,
            "do_mask_loss_for_padding": do_mask_loss_for_padding,
            "optimizer_lr": learning_rate,  # Map to LeRobot's parameter name
            "optimizer_betas": optimizer_betas,
            "optimizer_eps": optimizer_eps,
            "optimizer_weight_decay": optimizer_weight_decay,
            "scheduler_name": scheduler_name,
            "scheduler_warmup_steps": scheduler_warmup_steps,
            **kwargs,
        }

        self.learning_rate = learning_rate
        self._framework = "lerobot"

        # Policy will be initialized in setup()
        self._lerobot_policy: _LeRobotDiffusionPolicy | None = None
        self.model: nn.Module | None = None
        self._framework_policy: _LeRobotDiffusionPolicy | None = None

        self.save_hyperparameters()

    @property
    def lerobot_policy(self) -> _LeRobotDiffusionPolicy:
        """Get the initialized LeRobot policy.

        Returns:
            The initialized LeRobot Diffusion policy.

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

        # Get the underlying LeRobot dataset - handle both adapter and raw dataset
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif LeRobotDataset is not None and isinstance(train_dataset, LeRobotDataset):
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule for YAML configs."
            )
            raise TypeError(msg)

        features = dataset_to_policy_features(lerobot_dataset.meta.features)  # type: ignore[misc]
        stats = lerobot_dataset.meta.stats

        # Create LeRobot Diffusion configuration
        lerobot_config = _LeRobotDiffusionConfig(  # type: ignore[misc]
            input_features=features,
            output_features=features,
            **self._config_kwargs,
        )

        # Initialize the policy
        self._lerobot_policy = _LeRobotDiffusionPolicy(lerobot_config, dataset_stats=stats)  # type: ignore[arg-type,misc]
        self.model = self._lerobot_policy.diffusion
        self._framework_policy = self._lerobot_policy

    def forward(
        self,
        batch: dict[str, torch.Tensor] | Observation,
        *args: Any,  # noqa: ARG002, ANN401
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> torch.Tensor:
        """Forward pass delegates to LeRobot.

        Args:
            batch: A batch of data containing observations and actions (dict or Observation).
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The computed loss tensor.
        """
        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        loss, _ = self.lerobot_policy.forward(batch_dict)
        return loss

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation.

        Args:
            batch: A batch of data containing observations and actions.
            batch_idx: Index of the batch.

        Returns:
            The computed loss tensor.
        """
        del batch_idx  # Unused argument

        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        loss, _ = self.lerobot_policy.forward(batch_dict)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor] | GymObservation,
        batch_idx: int,
    ) -> torch.Tensor | dict[str, float]:
        """Validation step.

        Args:
            batch: A batch of data containing observations and actions, or GymObservation for gym validation.
            batch_idx: Index of the batch.

        Returns:
            The computed loss tensor for dataset validation, or metrics dict for gym validation.
        """
        # Check if this is gym validation - delegate to base class
        if isinstance(batch, GymObservation):
            return super().validation_step(batch, batch_idx)

        # Convert to LeRobot format if needed (handles Observation or collated dict)
        batch_dict = FormatConverter.to_lerobot_dict(batch)

        loss, _ = self.lerobot_policy.forward(batch_dict)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def select_action(self, batch: dict[str, torch.Tensor] | Observation) -> torch.Tensor:
        """Select action (inference mode) through LeRobot.

        Args:
            batch: A batch of data containing observations (dict or Observation).

        Returns:
            The selected action tensor.
        """
        batch_dict = FormatConverter.to_lerobot_dict(batch)
        return self.lerobot_policy.select_action(batch_dict)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's parameters.

        Returns:
            torch.optim.Optimizer: The configured optimizer instance.
        """
        return torch.optim.Adam(self.lerobot_policy.get_optim_params(), lr=self.learning_rate)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
