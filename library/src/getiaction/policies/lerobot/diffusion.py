# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Policy wrapper.

This module provides a Lightning-compatible wrapper around LeRobot's Diffusion Policy
implementation with explicit typed parameters for better IDE support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning_utilities.core.imports import module_available

from getiaction.policies.lerobot.universal import LeRobotPolicy

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

if module_available("lerobot"):
    LEROBOT_AVAILABLE = True
else:
    LEROBOT_AVAILABLE = False


class Diffusion(LeRobotPolicy):
    """Diffusion Policy from LeRobot.

    PyTorch Lightning wrapper around LeRobot's Diffusion Policy implementation that
    provides explicit typed parameters for better IDE support. Inherits all
    functionality from LeRobotPolicy.

    The Diffusion Policy uses denoising diffusion probabilistic models to generate
    robot actions through iterative denoising of Gaussian noise.

    Examples:
        Load pretrained model from HuggingFace Hub:

            >>> from getiaction.policies.lerobot import Diffusion
            >>> policy = Diffusion.from_pretrained("lerobot/diffusion_pusht")

        Create from dataset (eager initialization):

            >>> policy = Diffusion.from_dataset(
            ...     "lerobot/pusht",
            ...     n_obs_steps=2,
            ...     horizon=16,
            ... )

        Train from scratch with explicit arguments:

            >>> from getiaction.policies.lerobot import Diffusion
            >>> from getiaction.data.lerobot import LeRobotDataModule
            >>> import lightning as L

            >>> policy = Diffusion(
            ...     n_obs_steps=2,
            ...     horizon=16,
            ...     n_action_steps=8,
            ... )

            >>> datamodule = LeRobotDataModule(
            ...     repo_id="lerobot/pusht",
            ...     batch_size=64,
            ... )

            >>> trainer = L.Trainer(max_epochs=100)
            >>> trainer.fit(policy, datamodule)

        YAML configuration with LightningCLI:

            ```yaml
            model:
              class_path: getiaction.policies.lerobot.Diffusion
              init_args:
                n_obs_steps: 2
                horizon: 16
                n_action_steps: 8
                down_dims: [512, 1024, 2048]
            ```

    Note:
        This class provides explicit typed parameters for IDE autocomplete.
        For dynamic policy selection, use LeRobotPolicy directly.

    See Also:
        - LeRobotPolicy: Universal wrapper for any LeRobot policy
        - LeRobotDataModule: For loading LeRobot datasets
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
        optimizer_lr: float = 1e-4,
        optimizer_betas: tuple[float, float] = (0.95, 0.999),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 1e-6,
        # Scheduler
        scheduler_name: str = "cosine",
        scheduler_warmup_steps: int = 500,
        # Additional parameters via kwargs
        **kwargs: Any,
    ) -> None:
        """Initialize Diffusion policy wrapper.

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
            optimizer_lr: Learning rate for optimizer.
            optimizer_betas: Beta parameters for Adam optimizer.
            optimizer_eps: Epsilon for Adam optimizer.
            optimizer_weight_decay: Weight decay for optimizer.
            scheduler_name: Name of learning rate scheduler.
            scheduler_warmup_steps: Number of warmup steps for scheduler.
            **kwargs: Additional DiffusionConfig parameters.

        Raises:
            ImportError: If LeRobot is not installed.
        """
        if not LEROBOT_AVAILABLE:
            msg = "Diffusion requires LeRobot framework.\n\nInstall with:\n    pip install lerobot\n"
            raise ImportError(msg)

        super().__init__(
            policy_name="diffusion",
            n_obs_steps=n_obs_steps,
            horizon=horizon,
            n_action_steps=n_action_steps,
            drop_n_last_frames=drop_n_last_frames,
            vision_backbone=vision_backbone,
            crop_shape=crop_shape,
            crop_is_random=crop_is_random,
            pretrained_backbone_weights=pretrained_backbone_weights,
            use_group_norm=use_group_norm,
            spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
            use_separate_rgb_encoder_per_camera=use_separate_rgb_encoder_per_camera,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            use_film_scale_modulation=use_film_scale_modulation,
            noise_scheduler_type=noise_scheduler_type,
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            num_inference_steps=num_inference_steps,
            do_mask_loss_for_padding=do_mask_loss_for_padding,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            scheduler_name=scheduler_name,
            scheduler_warmup_steps=scheduler_warmup_steps,
            **kwargs,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: LeRobotDataset | str,
        **kwargs: Any,
    ) -> "Diffusion":
        """Create Diffusion policy with eager initialization from a dataset.

        Args:
            dataset: Either a LeRobotDataset instance or a HuggingFace Hub repo ID.
            **kwargs: Diffusion configuration parameters.

        Returns:
            Fully initialized Diffusion policy ready for inference.

        Examples:
            >>> policy = Diffusion.from_dataset("lerobot/pusht", horizon=16)
        """
        return LeRobotPolicy.from_dataset("diffusion", dataset, **kwargs)  # type: ignore[return-value]
