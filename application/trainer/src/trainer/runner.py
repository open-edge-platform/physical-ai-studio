# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training execution for the trainer service.

Trains with the `physicalai` Lightning trainer using a dataset snapshot
uploaded over HTTP, exports the policy, and zips the result for download.
torch/`physicalai` imports are deferred to call time.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from trainer.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch import LightningModule

    from trainer.schemas import SubmitJobRequest


ProgressFn = Callable[[int, str | None, dict[str, Any] | None], None]
StopFn = Callable[[], bool]


class JobCanceledError(Exception):
    """Raised when a job stops because cancellation was requested.

    Distinct from a genuine failure: the queue worker marks the job CANCELED and
    logs at info level instead of dumping an error traceback.
    """


class TrainerRunner:
    """Run a single training job end to end."""

    def run(self, job_id: str, request: SubmitJobRequest, *, should_stop: StopFn, report: ProgressFn) -> Path:
        """Execute training and return the path to the model archive."""
        settings = get_settings()
        snapshot_dir = settings.datasets_dir / job_id
        report(0, "Dataset ready", None)

        model_dir = settings.models_dir / job_id
        cache_dir = settings.storage_dir / "cache" / job_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._train(request, snapshot_dir, model_dir, cache_dir, should_stop=should_stop, report=report)
        finally:
            self._cleanup_uploaded_dataset(job_id)

        report(100, "Archiving model", None)
        return self._archive_model(job_id, model_dir)

    @staticmethod
    def _cleanup_uploaded_dataset(job_id: str) -> None:
        """Remove the uploaded dataset once the job no longer needs it."""
        dataset_dir = get_settings().datasets_dir / job_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

    def _train(
        self,
        request: SubmitJobRequest,
        snapshot_dir: Path,
        model_dir: Path,
        cache_dir: Path,
        *,
        should_stop: StopFn,
        report: ProgressFn,
    ) -> None:
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from physicalai.data import LeRobotDataModule
        from physicalai.train import ProgressReportingCallback, Trainer

        payload = request.payload
        accelerator, strategy, devices = self._resolve_device()

        data_module = LeRobotDataModule(
            repo_id="snapshot",
            root=str(snapshot_dir),
            train_batch_size=int(payload.get("batch_size", 8)),
            num_workers=payload.get("num_workers", "auto"),
            val_split=float(payload.get("val_split", 0.1)),
        )

        policy = self._setup_policy(request.policy, compile_model=bool(payload.get("compile_model", False)))

        checkpoint_callback = ModelCheckpoint(
            dirpath=cache_dir,
            filename="model",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
        )
        csv_logger = CSVLogger(cache_dir.parent, name=cache_dir.stem)

        trainer = Trainer(
            logger=csv_logger,
            callbacks=[checkpoint_callback, ProgressReportingCallback(report=report, should_stop=should_stop)],
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            max_steps=int(payload.get("max_steps", 100)),
            auto_scale_batch_size=bool(payload.get("auto_scale_batch_size", False)),
            precision=str(payload.get("precision", "bf16-mixed")),
            check_val_every_n_epoch=1,
        )

        report(0, "Training model", None)
        trainer.fit(model=policy, datamodule=data_module)
        if should_stop():
            msg = "Training canceled"
            raise JobCanceledError(msg)

        trainer.save_checkpoint(cache_dir / "model.ckpt")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(cache_dir), str(model_dir))

        self._export_policy(policy, model_dir, report)

    def _export_policy(self, policy: object, model_dir: Path, report: ProgressFn) -> None:
        from physicalai.export import ExportablePolicyMixin

        if not isinstance(policy, ExportablePolicyMixin):
            logger.info("Policy does not support export backends; skipping export")
            return

        for backend in policy.get_supported_export_backends():
            backend_name = backend.value if hasattr(backend, "value") else str(backend)
            try:
                report(100, f"Exporting to {backend_name}", None)
                policy.export(model_dir / "exports" / backend_name, backend=backend)
            except ImportError as exc:
                # An optional backend dependency is not installed (e.g. executorch
                # on xpu builds). Skip it without a traceback so the job isn't
                # mistaken for a failure; other backends still export.
                logger.warning("Skipping {} export: optional dependency missing ({})", backend_name, exc)
            except Exception as exc:  # noqa: BLE001  # export is best-effort; one backend failing must not abort the job
                logger.error("Export to {} failed", backend_name)
                logger.exception(exc)

    @staticmethod
    def _setup_policy(policy_name: str, *, compile_model: bool) -> LightningModule:
        from physicalai.policies import ACT, Pi0, Pi05, SmolVLA

        if policy_name == "act":
            return ACT(compile_model=compile_model)
        if policy_name == "pi0":
            return Pi0(compile_model=compile_model)
        if policy_name == "pi05":
            return Pi05(pretrained_name_or_path="lerobot/pi05_base", compile_model=compile_model)
        if policy_name == "smolvla":
            return SmolVLA(pretrained_name_or_path="lerobot/smolvla_base", compile_model=compile_model)
        msg = f"Policy not implemented: {policy_name}"
        raise ValueError(msg)

    @staticmethod
    def _resolve_device() -> tuple[str, str, list[int] | int]:
        import torch

        if torch.xpu.is_available():
            return "xpu", "xpu_single", 1
        if torch.cuda.is_available():
            return "cuda", "auto", 1
        return "cpu", "auto", 1

    def _archive_model(self, job_id: str, model_dir: Path) -> Path:
        archives_dir = get_settings().archives_dir
        archives_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archives_dir / f"{job_id}.zip"
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in model_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(model_dir))
        return archive_path
