# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""In-process training backend using torch/Lightning.

Imports of `physicalai`, torch, and Lightning are deferred to call time so this
module can be imported in environments without the `[train]` extra installed.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from loguru import logger

from models.utils import load_policy, setup_policy
from services.training_backends._log_format import render_progress_log
from utils.device import get_lightning_strategy, get_torch_device

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.callbacks import Callback

    from services.training_backends.base import TrainingContext


class LocalTrainingBackend:
    """Train in the worker process with Lightning."""

    async def train(self, context: TrainingContext) -> None:
        """Run Lightning training, save, and export into the model directory."""
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from physicalai.data import LeRobotDataModule
        from physicalai.train import Trainer

        payload = context.payload
        output_dir = context.output_dir
        cache_path = context.cache_dir

        if context.snapshot is None:
            raise ValueError("Local training requires a dataset snapshot")

        device_type = payload.device.type if payload.device else None
        device_index = payload.device.index if payload.device else None
        accelerator = get_torch_device(device_type)

        l_dm = LeRobotDataModule(
            repo_id="snapshot",  # irrelevant for loading from a local root
            root=context.snapshot.path,
            train_batch_size=payload.batch_size,
            num_workers=payload.num_workers,
            val_split=payload.val_split,
        )

        if context.base_model is not None:
            policy = load_policy(context.base_model, compile_model=payload.compile_model)
        else:
            policy = setup_policy(context.model, compile_model=payload.compile_model)

        precision = str(payload.precision)
        strategy = get_lightning_strategy(device_type)
        devices = [device_index] if device_index is not None else 1

        checkpoint_callback = ModelCheckpoint(
            dirpath=cache_path,
            filename="model",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
        )
        csv_logger = CSVLogger(cache_path.parent, name=cache_path.stem)

        trainer = Trainer(
            logger=csv_logger,
            callbacks=[
                checkpoint_callback,
                self._progress_callback(context),
            ],
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            max_steps=payload.max_steps,
            auto_scale_batch_size=payload.auto_scale_batch_size,
            precision=precision,
            check_val_every_n_epoch=1,
        )

        trainer.fit(model=policy, datamodule=l_dm)

        final_checkpoint = cache_path / "model.ckpt"
        trainer.save_checkpoint(final_checkpoint)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cache_path), str(output_dir))

        export_policy = policy
        if payload.compile_model and context.model.policy in ("act", "smolvla"):
            try:
                logger.info("Reloading non-compiled policy for export")
                export_policy = load_policy(context.model, compile_model=False)
            except Exception as exc:
                logger.warning("Failed to reload non-compiled policy for export; using trained policy")
                logger.exception(exc)

        self._export_policy(policy=export_policy, output_dir=output_dir, context=context)

    def _progress_callback(self, context: TrainingContext) -> Callback:
        """Build the shared progress callback wired to this job's reporter.

        Reuses `physicalai.train.ProgressReportingCallback` so local runs emit
        the same telemetry as remote ones. The reporter both mirrors loggable
        telemetry to the job log (via the shared renderer) and updates job
        progress, reserving 100% for the terminal completion update.
        """
        from physicalai.train import ProgressReportingCallback

        reporter = context.progress

        def report(progress: int, message: str | None, extra_info: dict) -> None:
            line = render_progress_log(extra_info)
            if line is not None:
                logger.info(line)
            # Cap running progress at 99; the worker writes 100 on completion.
            reporter(min(99, progress), message=message, extra_info=extra_info)

        return ProgressReportingCallback(report=report, should_stop=context.should_stop)

    def _export_policy(self, *, policy: object, output_dir: Path, context: TrainingContext) -> None:
        """Export the trained policy to every backend the policy supports."""
        from physicalai.export import ExportablePolicyMixin

        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        logger.info("Starting model export for trained policy")
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            try:
                logger.info("Exporting model to {} format", backend_name)
                context.progress(99, message=f"Exporting to {backend_name} format")
                export_dir = output_dir / "exports" / backend_name
                policy.export(export_dir, backend=backend)
                logger.info("Model export to {} completed", backend_name)
            except ImportError as exc:
                # Optional backend dependency not installed; skip without a
                # traceback so the run isn't mistaken for a failure.
                logger.warning("Skipping {} export: optional dependency missing ({})", backend_name, exc)
            except Exception as exc:
                logger.error("Failed exporting model to {} format", backend_name)
                logger.exception(exc)
