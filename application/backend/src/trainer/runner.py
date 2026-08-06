# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training execution for the trainer service.

Delegates the training run itself to :func:`training.run_training_job`, the
same call the studio uses for in-process training, so a policy trains
identically here and there. This module owns only what is specific to serving
jobs remotely: where the uploaded snapshot lives, cleaning it up afterwards, and
zipping the result for download.
"""

from __future__ import annotations

import shutil
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trainer.settings import get_settings

if TYPE_CHECKING:
    from pathlib import Path

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

    @staticmethod
    def cleanup_job_outputs(job_id: str) -> None:
        """Remove a job's model output and checkpoint cache from disk.

        Called for jobs that end up FAILED or CANCELED: they have no artifact
        worth keeping, so nothing should be left behind on the trainer's disk.
        A job canceled mid-training in particular can leave its checkpoint
        cache directory unmoved since the move only happens on a successful
        finish, so this must clean up both directories.
        """
        settings = get_settings()
        for path in (settings.models_dir / job_id, settings.storage_dir / "cache" / job_id):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _train(
        request: SubmitJobRequest,
        snapshot_dir: Path,
        model_dir: Path,
        cache_dir: Path,
        *,
        should_stop: StopFn,
        report: ProgressFn,
    ) -> None:
        """Run the shared training job and translate cancellation into an error.

        ``run_training_job`` reports a canceled run by returning without an
        artifact; the queue worker distinguishes CANCELED from FAILED by
        exception type, so the cancellation is re-raised here.
        """
        from training import run_training_job

        run_training_job(
            request.spec,
            dataset_root=snapshot_dir,
            output_dir=model_dir,
            cache_dir=cache_dir,
            report=report,
            should_stop=should_stop,
        )

        if should_stop():
            msg = "Training canceled"
            raise JobCanceledError(msg)

    @staticmethod
    def _archive_model(job_id: str, model_dir: Path) -> Path:
        archives_dir = get_settings().archives_dir
        archives_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archives_dir / f"{job_id}.zip"
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in model_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(model_dir))
        return archive_path
