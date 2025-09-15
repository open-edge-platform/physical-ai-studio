# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning Callbacks for Action Training"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import imageio
import numpy as np
from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    import lightning


class ActionMetricsCallback(Callback):
    """
    A callback to aggregate and log rollout metrics for validation and test stages
    without using on_epoch_end hooks.
    """

    def __init__(self, stage: str):
        """
        Args:
            stage (str): Either 'val' or 'test' to indicate which stage to log.
        """
        super().__init__()
        assert stage in ["val", "test"], "stage must be 'val' or 'test'"
        self.stage = stage

    def on_validation_batch_end(
        self,
        trainer: lightning.Trainer,
        lightning_module: lightning.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.stage != "val":
            return
        self._log_metrics(lightning_module, outputs)

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        lightning_module: lightning.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.stage != "test":
            return
        self._log_metrics(lightning_module, outputs)

    def _log_metrics(self, lightning_module, outputs):
        """
        Aggregates metrics from a single rollout/batch and logs them.
        """
        if outputs is None:
            return

        # Aggregate metrics
        success = float(outputs.get("success", 0))
        steps = float(outputs.get("steps", 0))
        sum_reward = float(outputs.get("sum_reward", 0))
        max_reward = float(outputs.get("max_reward", 0))
        inference_times = outputs.get("inference_times", [])
        mean_inference_time = float(np.mean(inference_times)) if inference_times else 0.0

        # Log metrics
        lightning_module.log(
            f"{self.stage}/success",
            success,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )
        lightning_module.log(
            f"{self.stage}/mean_steps",
            steps,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )
        lightning_module.log(
            f"{self.stage}/avg_sum_reward",
            sum_reward,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )
        lightning_module.log(
            f"{self.stage}/avg_max_reward",
            max_reward,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )
        lightning_module.log(
            f"{self.stage}/mean_inference_time",
            mean_inference_time,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )


class TestVideoLogger(Callback):
    """A PyTorch Lightning callback to log videos during the test phase.

    This callback captures the frames from the test rollout and saves them
    as a video file.
    """

    def on_test_batch_end(
        self,
        trainer: lightning.Trainer,
        lightning_module: lightning.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called when a test batch ends.

        This method extracts video frames from the outputs and saves them
        to a video file.

        Args:
            trainer (L.Trainer): The PyTorch Lightning trainer instance.
            lightning_module (L.LightningModule): The PyTorch Lightning module instance.
            outputs (dict): The outputs from the `test_step`. It should
                contain a "frames" key.
            batch (dict): The test batch. It should contain an "env" key.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the dataloader.
        """
        video_buffer = np.stack(outputs["frames"])  # (T, C, H, W)
        fps = batch["env"].render_fps

        # Guess the output directory from the trainer's logger or a default.
        if trainer.logger and trainer.logger.log_dir:
            output_dir = Path(trainer.logger.log_dir) / "test_videos"
        else:
            output_dir = Path("test_videos")

        # Ensure the output directory exists
        output_dir.mkdir(exist_ok=True)

        # Define the output file path
        output_path = output_dir / f"rollout_{batch_idx}.mp4"

        # Convert the video buffer to (T, H, W, C) for imageio
        video_buffer = np.transpose(video_buffer, (0, 2, 3, 1))

        # Handle the case where the image format is not uint8
        if video_buffer.dtype != np.uint8:
            video_buffer = (video_buffer * 255).astype(np.uint8)

        imageio.mimsave(output_path, video_buffer, fps=fps)
