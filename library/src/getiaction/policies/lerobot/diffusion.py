# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's Diffusion Policy implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from getiaction.policies.base import Policy

if TYPE_CHECKING:
    from torch import nn

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as _LeRobotDiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as _LeRobotDiffusionPolicy

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    _LeRobotDiffusionPolicy = None
    _LeRobotDiffusionConfig = None


class Diffusion(Policy):
    """Diffusion Policy from LeRobot with lazy initialization.

    The LeRobot policy is created in setup() after the dataset is loaded.
    This enables YAML configuration while maintaining compatibility with Python scripts.

    Example YAML usage:
        model:
          class_path: getiaction.policies.lerobot.Diffusion
          init_args:
            n_obs_steps: 2
            horizon: 16
            n_action_steps: 8
        data:
          class_path: getiaction.data.lerobot.LeRobotDataModule
          init_args:
            repo_id: lerobot/pusht

    Example Python usage:
        policy = Diffusion(n_obs_steps=2, horizon=16, n_action_steps=8)
        datamodule = LeRobotDataModule(repo_id="lerobot/pusht")
        trainer.fit(policy, datamodule)  # setup() called automatically
    """

    def __init__(
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
        **kwargs: Any,
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
        """
        if not LEROBOT_AVAILABLE:
            msg = "Diffusion requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
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
        self.lerobot_policy: _LeRobotDiffusionPolicy | None = None
        self.model: nn.Module | None = None
        self._framework_policy: _LeRobotDiffusionPolicy | None = None

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """Set up the policy from datamodule if not already initialized.

        This method is called by Lightning before fit/validate/test/predict.
        It extracts features from the datamodule's training dataset and
        initializes the policy if it wasn't already created in __init__.

        Args:
            stage: The stage of training ('fit', 'validate', 'test', or 'predict')
        """
        if self.lerobot_policy is not None:
            return  # Already initialized

        # Extract features from datamodule
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.utils import dataset_to_policy_features

        from getiaction.data.lerobot import _LeRobotDatasetAdapter

        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both adapter and raw dataset
        if isinstance(train_dataset, _LeRobotDatasetAdapter):
            lerobot_dataset = train_dataset._lerobot_dataset  # noqa: SLF001
        elif isinstance(train_dataset, LeRobotDataset):
            lerobot_dataset = train_dataset
        else:
            msg = (
                f"Expected train_dataset to be _LeRobotDatasetAdapter or LeRobotDataset, "
                f"got {type(train_dataset)}. Use LeRobotDataModule for YAML configs."
            )
            raise TypeError(msg)

        features = dataset_to_policy_features(lerobot_dataset.meta.features)
        stats = lerobot_dataset.meta.stats

        # Create LeRobot Diffusion configuration
        lerobot_config = _LeRobotDiffusionConfig(
            input_features=features,
            output_features=features,
            **self._config_kwargs,
        )

        # Initialize the policy
        self.lerobot_policy = _LeRobotDiffusionPolicy(lerobot_config, dataset_stats=stats)
        self.model = self.lerobot_policy.diffusion
        self._framework_policy = self.lerobot_policy

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass delegates to LeRobot."""
        loss, _ = self.lerobot_policy.forward(batch)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step uses LeRobot's loss computation."""
        loss, _ = self.lerobot_policy.forward(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, _ = self.lerobot_policy.forward(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action (inference mode) through LeRobot."""
        return self.lerobot_policy.select_action(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer using LeRobot's parameters."""
        return torch.optim.Adam(self.lerobot_policy.get_optim_params(), lr=self.learning_rate)

    def reset(self) -> None:
        """Reset the policy state."""
        self.lerobot_policy.reset()
