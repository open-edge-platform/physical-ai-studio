# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset-transfer strategies for the remote training backend.

Each strategy owns how the dataset snapshot reaches the trainer (streamed over
HTTP or pushed via an ephemeral HuggingFace repo) and how that transfer is
cleaned up. The shared "wait for completion and ingest the model" tail lives on
:class:`~services.training_backends.remote.RemoteTrainingBackend` so both
strategies stay focused on transport.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from services.archive_safety import cleanup_staged_archive
from services.dataset_download_service import DatasetDownloadService
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
        archive_path: Path | None = None
        try:
            # Sub-step 1: zip the snapshot and stream it to the trainer (0-10%).
            context.progress(0, message="Preparing dataset snapshot")
            archive = await asyncio.to_thread(
                DatasetDownloadService().create_dataset_archive, Path(context.snapshot.path)
            )
            archive_path = archive
            remote_job_id = await backend.submit_job(context, dataset_transfer="http")
            await backend.upload_snapshot_http(context, remote_job_id, archive)
            context.progress(SNAPSHOT_UPLOAD_PROGRESS, message="Dataset uploaded, starting training")

            # Sub-steps 2 & 3: wait for the remote job, then ingest the model.
            await backend.await_and_ingest(context, remote_job_id)
        finally:
            if archive_path is not None:
                cleanup_staged_archive(archive_path)


class HfTrainingMethod(TrainingMethod):
    """Push the snapshot to an ephemeral private HuggingFace dataset repo."""

    async def train(self, context: TrainingContext) -> None:
        backend = self._backend
        repo_id: str | None = None
        try:
            # Sub-step 1: push the snapshot to an ephemeral private dataset repo (0-10%).
            context.progress(0, message="Uploading dataset snapshot")
            repo_id, revision = await backend.push_snapshot(context)
            context.progress(SNAPSHOT_UPLOAD_PROGRESS, message="Dataset uploaded, starting training")

            # Sub-step 1b: submit the job now that the snapshot repo exists.
            remote_job_id = await backend.submit_job(context, dataset_transfer="hf", repo_id=repo_id, revision=revision)

            # Sub-steps 2 & 3: wait for the remote job, then ingest the model.
            await backend.await_and_ingest(context, remote_job_id)
        finally:
            if repo_id is not None:
                await backend.delete_repo(repo_id)
