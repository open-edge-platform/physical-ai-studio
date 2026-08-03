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

        # Step-based checkpoints. The previous val/loss-monitored callback never
        # fired on short runs: one epoch is ~11k steps at batch 8 on the current
        # dataset, so a sub-epoch run completed no validation and saved nothing
        # until the explicit save below. Saving on a step interval instead means
        # an unattended run always has a recent state on disk.
        checkpoint_callback = ModelCheckpoint(
            dirpath=cache_path,
            filename="step{step:06d}",
            # Keep this a divisor of the max_steps values actually used (1000 /
            # 3000 / 5000 / 12000), so the *final* step always lands on an
            # interval and gets a checkpoint from this callback. A coarser
            # interval like 3000 would miss step 5000 entirely and leave the
            # run depending on the explicit save below, which is the step that
            # OOMed a completed 5000-step run on 2026-07-26.
            every_n_train_steps=1000,
            # save_top_k=-1 keeps every checkpoint. With monitor=None Lightning
            # only accepts -1, 0 or 1, since there is no metric to rank by.
            save_top_k=-1,
            monitor=None,
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
            # Guards against the loss divergence seen on a 4k-step bf16 run,
            # where training went to NaN mid-run and kept going.
            gradient_clip_val=1.0,
            # Validate on a step interval as well, so sub-epoch runs report
            # val/loss instead of only a single value at the very end.
            val_check_interval=1000,
            check_val_every_n_epoch=1,
        )

        trainer.fit(model=policy, datamodule=l_dm)

        final_checkpoint = cache_path / "model.ckpt"
        trainer.save_checkpoint(final_checkpoint)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cache_path), str(output_dir))

        # Release the training-side memory before exporting. A completed
        # 5000-step run (`9200784f`, 2026-07-26) was OOM-killed here: the host
        # has 30 GB, the process held ~26 GB of trainer/dataloader state, and
        # ONNX/OpenVINO conversion needs several GB more on top. An OOM kill is
        # SIGKILL, so no try/except downstream can rescue it — the only fix is
        # to hand the memory back first.
        del trainer, l_dm
        self._release_memory(accelerator)

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

    def _release_memory(self, accelerator: str | None = None) -> None:
        """Drop cached host and device memory between training and export."""
        import gc

        gc.collect()
        try:
            import torch

            if accelerator == "xpu" and torch.xpu.is_available():
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Could not release device cache: {}", exc)

    def _export_policy(self, *, policy: object, output_dir: Path, context: TrainingContext) -> None:
        """Export the trained policy to every backend the policy supports."""
        import os

        from physicalai.export import ExportablePolicyMixin

        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Skipping export: policy does not support export backends")
            return

        # OpenVINO conversion is the most memory-hungry backend and the one that
        # has actually killed runs here: `ec4c4cef` (2000 steps, 2026-07-25) died
        # part-way through OpenVINO's ONNX initializer pass with the trained
        # weights on disk but no model registered. Torch export alone is enough
        # to load a model in the GUI, so OpenVINO is opt-in via
        # PHYSICALAI_EXPORT_BACKENDS (comma-separated, e.g. "torch,openvino").
        # Default keeps the cheap, proven backend only.
        requested = os.environ.get("PHYSICALAI_EXPORT_BACKENDS", "torch")
        wanted = {b.strip().lower() for b in requested.split(",") if b.strip()}

        logger.info("Starting model export for trained policy (backends requested: {})", requested)
        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            if backend_name.lower() not in wanted:
                logger.info(
                    "Skipping {} export: not in PHYSICALAI_EXPORT_BACKENDS ({}). "
                    "Export it later with local-changes/export-rescued-model.py",
                    backend_name,
                    requested,
                )
                continue
            # Conversion peaks well above the model's resident size; hand back
            # whatever the previous backend cached before starting the next one.
            self._release_memory()
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
