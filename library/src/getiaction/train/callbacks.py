# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Callbacks for training."""

from pathlib import Path
from typing import Any

import imageio
import lightning as L  # noqa: N812
import torch
from lightning.pytorch.callbacks import Callback

from getiaction.data import Observation
from getiaction.gyms import Gym
from getiaction.train.utils import reformat_dataset_to_match_policy


class PolicyDatasetInteraction(Callback):
    """Callback to interact the policy and dataset before training starts."""

    @staticmethod
    def _interact_policy_dataset(trainer: L.Trainer, model: L.LightningModule) -> None:
        # Assumes trainer has a datamodule attached
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self._interact_policy_dataset(trainer, pl_module)


class VideoLogger(L.Callback):
    """Callback to log videos during PyTorch Lightning training/testing.

    This callback logs video frames from observations during different phases of training.
    For training/validation: extracts video from Observation batches
    For testing: logs rollout videos from Gym environments during evaluation
    Videos are saved to the filesystem in MP4 format with organized directory structure.

    Args:
        output_dir: Directory to save videos
        fps: Frames per second for saved videos
        log_every_n_batches: Save video every N batches (default: 1)
        max_videos_per_phase: Maximum videos to save per phase (default: 5)
        phases: Which phases to log videos for (default: ["test"])
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = 30,
        log_every_n_batches: int = 1,
        max_videos_per_phase: int = 5,
        phases: list[str] | None = None,
    ) -> None:
        """Initialize VideoLogger."""
        super().__init__()
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.log_every_n_batches = log_every_n_batches
        self.max_videos_per_phase = max_videos_per_phase
        self.phases = phases or ["test"]

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track video counts per phase
        self.video_counts = {phase: 0 for phase in self.phases}

        # Store rollout frames during test episodes
        self.current_rollout_frames = []

    def _extract_video_from_observation(self, batch: Observation) -> torch.Tensor | None:
        """Extract video frames from Observation batch.

        Args:
            batch: Observation batch containing images

        Returns:
            Video frames tensor or None if no video found
        """
        if batch.images is None:
            return None

        # Handle different image formats
        if isinstance(batch.images, dict):
            # Multiple cameras - pick first available
            for camera_name, frames in batch.images.items():
                if isinstance(frames, torch.Tensor) and frames.ndim >= 3:
                    return frames
        elif isinstance(batch.images, torch.Tensor) and batch.images.ndim >= 3:
            return batch.images

        return None

    def _save_video(self, frames: torch.Tensor, filename: str) -> None:
        """Save video frames to file.

        Args:
            frames: Video frames tensor [B, H, W, C] or [B, C, H, W] or [T, H, W, C]
            filename: Output filename
        """
        # Convert to numpy and ensure correct format
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()

        # Handle different tensor formats
        if frames.ndim == 4:
            # If channels first [B/T, C, H, W], transpose to [B/T, H, W, C]
            if frames.shape[1] <= 4:  # Assume channels first if small second dimension
                frames = frames.transpose(0, 2, 3, 1)
        elif frames.ndim == 3:
            # Single frame [H, W, C] or [C, H, W]
            if frames.shape[0] <= 4:  # Channels first
                frames = frames.transpose(1, 2, 0)
            frames = frames[None, ...]  # Add batch dimension

        # Ensure values are in [0, 255] range for video
        if frames.max() <= 1.0:
            frames = (frames * 255).astype('uint8')
        else:
            frames = frames.astype('uint8')

        # Save video
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with imageio.get_writer(str(filepath), fps=self.fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    def _should_log_video(self, phase: str, batch_idx: int) -> bool:
        """Check if we should log video for this batch.

        Args:
            phase: Current phase (train/val/test)
            batch_idx: Current batch index

        Returns:
            True if should log video
        """
        if phase not in self.phases:
            return False

        if self.video_counts[phase] >= self.max_videos_per_phase:
            return False

        if batch_idx % self.log_every_n_batches != 0:
            return False

        return True

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Observation,
        batch_idx: int,
    ) -> None:
        """Called after training batch ends."""
        if not self._should_log_video("train", batch_idx):
            return

        frames = self._extract_video_from_observation(batch)
        if frames is not None:
            epoch = trainer.current_epoch
            filename = f"train/epoch_{epoch:03d}_batch_{batch_idx:04d}.mp4"
            self._save_video(frames, filename)
            self.video_counts["train"] += 1

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Observation,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called after validation batch ends."""
        if not self._should_log_video("val", batch_idx):
            return

        frames = self._extract_video_from_observation(batch)
        if frames is not None:
            epoch = trainer.current_epoch
            filename = f"val/epoch_{epoch:03d}_batch_{batch_idx:04d}.mp4"
            self._save_video(frames, filename)
            self.video_counts["val"] += 1

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Gym,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called after test batch ends.

        For test phase, batch is actually a Gym environment.
        The rollout evaluation happens inside the test_step, and we can
        capture the video frames from the environment if available.
        """
        if not self._should_log_video("test", batch_idx):
            return

        # For gym-based testing, we would need to modify the rollout evaluation
        # to capture frames. This is a placeholder for that functionality.
        # The actual implementation would require coordination with the rollout
        # evaluation code to capture observation frames during the episode.

        # TODO: Implement gym rollout video capture
        # This would require modifying the evaluate_gym method to collect frames
        # during the rollout and make them available here.
        pass
