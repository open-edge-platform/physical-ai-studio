# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""One-off training run for batch workloads (AWS Batch today).

The container runs exactly one job and exits. Everything about the job lives
under one storage prefix: Studio writes ``spec.json`` and ``dataset.zip``;
the container writes ``status.json`` while training and ``artifact.zip`` at
the end. :class:`~trainer.batch.storage.JobStorage` is the only
provider-specific seam, so this module and the training logic in
:func:`training.run_training_job` stay cloud-agnostic.
"""

from __future__ import annotations

import shutil
import signal
import threading

from loguru import logger

from trainer.archive import validate_and_extract
from trainer.batch.settings import BatchSettings
from trainer.batch.status import BatchJobStatus, StatusReporter
from trainer.batch.storage import JobStorage, LocalDirJobStorage
from trainer.runner import archive_model
from trainer.schemas import TrainerJobStatus
from trainer.settings import get_settings
from training import RunOptions

__all__ = [
    "EXIT_CANCELED",
    "EXIT_FAILED",
    "EXIT_OK",
    "BatchJobStatus",
    "BatchSettings",
    "JobStorage",
    "LocalDirJobStorage",
    "StatusReporter",
    "install_sigterm_handler",
    "run",
]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CANCELED = 130


def install_sigterm_handler(stop: threading.Event) -> None:
    """Translate the scheduler's termination signal into a cooperative stop request."""

    def _on_sigterm(_signum: int, _frame: object) -> None:
        logger.info("Received SIGTERM; stopping training")
        stop.set()

    signal.signal(signal.SIGTERM, _on_sigterm)


def run(storage: JobStorage, settings: BatchSettings, *, stop: threading.Event | None = None) -> int:
    """Fetch inputs, train, publish outputs. Returns the process exit code."""
    stop = stop or threading.Event()
    reporter = StatusReporter(storage, settings.status_interval_s)
    trainer_settings = get_settings()
    work_dir = trainer_settings.storage_dir / "batch" / settings.job_id
    archive_path = work_dir / "dataset.zip"
    dataset_dir = work_dir / "dataset"
    model_dir = trainer_settings.models_dir / settings.job_id
    cache_dir = trainer_settings.storage_dir / "cache" / settings.job_id

    try:
        reporter.finish(TrainerJobStatus.RUNNING, progress=0, message="Fetching dataset")
        spec = storage.fetch_spec()
        # run_options are runner-local; whatever Studio serialised (e.g. a
        # masked token) must not leak into this run.
        spec.run_options = RunOptions()
        storage.fetch_dataset(archive_path)
        validate_and_extract(archive_path, dataset_dir)
        archive_path.unlink(missing_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        reporter.finish(TrainerJobStatus.RUNNING, progress=0, message="Dataset ready")

        from training import run_training_job

        run_training_job(
            spec,
            dataset_root=dataset_dir,
            output_dir=model_dir,
            cache_dir=cache_dir,
            report=reporter.report,
            should_stop=stop.is_set,
        )
        if stop.is_set():
            reporter.finish(TrainerJobStatus.CANCELED, progress=0, message="Training canceled")
            return EXIT_CANCELED

        model_archive = archive_model(settings.job_id, model_dir, report=reporter.report)
        storage.put_artifact(model_archive)
        reporter.finish(TrainerJobStatus.COMPLETED, progress=100, message="Model uploaded")
        return EXIT_OK
    except Exception as exc:
        logger.opt(exception=True).error("Batch training failed")
        reporter.finish(TrainerJobStatus.FAILED, progress=0, message=str(exc))
        return EXIT_FAILED
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
