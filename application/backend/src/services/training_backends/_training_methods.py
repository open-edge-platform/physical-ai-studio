# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset-transfer strategies for the remote training backend."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from services.dataset_download_service import DatasetDownloadService
from services.staged_archive import cleanup_staged_archive
from services.training_backends.remote import SNAPSHOT_UPLOAD_PROGRESS

if TYPE_CHECKING:
    from services.training_backends.base import TrainingContext
    from services.training_backends.remote import RemoteTrainingBackend


class TrainingMethod(ABC):
    """Strategy for delivering the dataset snapshot to the trainer and running the job."""

    def __init__(self, backend: RemoteTrainingBackend) -> None:
        self._backend = backend

    @abstractmethod
    async def train(self, context: TrainingContext) -> None:
        """Deliver the snapshot, submit the job, mirror progress, ingest the model."""


class HttpTrainingMethod(TrainingMethod):
    """Stream the snapshot ZIP straight to the trainer over HTTP."""

    async def train(self, context: TrainingContext) -> None:
        backend = self._backend
        if context.snapshot is None:
            raise ValueError("HTTP dataset transfer requires a dataset snapshot")

        # Archive and upload the snapshot (0-10%).
        context.progress(0, message="Preparing dataset snapshot")
        archive_path = await asyncio.to_thread(
            DatasetDownloadService().create_dataset_archive, Path(context.snapshot.path)
        )
        try:
            remote_job_id = await backend.submit_job(context)

            # Persist before the upload begins so a restart can resume it.
            if context.on_remote_job_id is not None:
                await context.on_remote_job_id(remote_job_id)

            await backend.upload_snapshot_http(context, remote_job_id, archive_path)
            context.progress(SNAPSHOT_UPLOAD_PROGRESS, message="Dataset uploaded, starting training")

            await backend.await_and_ingest(context, remote_job_id)
        finally:
            cleanup_staged_archive(archive_path)
