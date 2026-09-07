# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SSH-provisioned training backend.

Provisions a per-job trainer container on a configured remote server, then
delegates dataset transfer, progress mirroring, and model ingestion to
:class:`~services.training_backends.remote.RemoteTrainingBackend` over the SSH
tunnel `services.ssh.provisioning` opened. The container and tunnel are always
torn down before this backend returns or raises, except when the studio is
shutting down (`TrainingSuspendedError`): that path deliberately leaves the
container running so a restart can reattach to it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Final

from loguru import logger

from db import get_async_db_session_ctx
from repositories.job_provisioning_repo import JobProvisioningRepository
from schemas.job import TrainingDevice
from services.ssh.preflight import DEFAULT_PROTOCOL_VERSION
from services.ssh.provisioning import ProvisionedTrainer, SshProvisioningService
from services.training_backends.base import TrainingCanceledError, TrainingSuspendedError
from services.training_backends.phase import SSH_PHASE_WINDOWS, PhaseKey, PhaseState, report_phase
from services.training_backends.remote import (
    SNAPSHOT_UPLOAD_PROGRESS,
    TRAINING_PROGRESS_END,
    RemoteTrainingBackend,
    RemoteTrainingError,
)
from settings import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from schemas.remote_server import RemoteServer
    from services.training_backends.base import ProgressReporter, TrainingContext

# Trainer containers only ever run on cuda/xpu servers (see
# `schemas.remote_server.SSH_SERVER_DEVICE_TYPES`), so the accelerator always
# needs an index.
_INDEXED_DEVICE_INDEX: Final = 0


def _directory_size_bytes(path: Path) -> int:
    """Sum the size of every regular file under ``path`` (blocking)."""
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _map_remote_progress_to_phase(raw_progress: int) -> tuple[PhaseKey, float]:
    """Map `RemoteTrainingBackend`'s own 0-100 progress into an SSH phase key + local completion.

    `RemoteTrainingBackend` partitions 0-100 into upload/train/download using
    its own constants, unaware that an SSH-provisioned job reserves a smaller
    slice of the overall bar for those same three steps. This translates one
    partition into the other.
    """
    if raw_progress < SNAPSHOT_UPLOAD_PROGRESS:
        return PhaseKey.UPLOAD, raw_progress / SNAPSHOT_UPLOAD_PROGRESS * 100
    if raw_progress < TRAINING_PROGRESS_END:
        span = TRAINING_PROGRESS_END - SNAPSHOT_UPLOAD_PROGRESS
        return PhaseKey.TRAIN, (raw_progress - SNAPSHOT_UPLOAD_PROGRESS) / span * 100
    span = 100 - TRAINING_PROGRESS_END
    if span <= 0:
        return PhaseKey.DOWNLOAD, 100.0
    return PhaseKey.DOWNLOAD, (raw_progress - TRAINING_PROGRESS_END) / span * 100


class _PhaseTaggingProgress:
    """Wrap a job's real `ProgressReporter` so `RemoteTrainingBackend`'s plain
    upload/train/download progress is remapped into the SSH phase table
    before it reaches the job store, without `RemoteTrainingBackend` itself
    knowing an SSH phase table exists.
    """

    def __init__(self, progress: ProgressReporter) -> None:
        self._progress = progress

    def __call__(self, progress: int, *, message: str | None = None, extra_info: dict | None = None) -> None:
        key, sub_progress = _map_remote_progress_to_phase(progress)
        report_phase(
            self._progress,
            SSH_PHASE_WINDOWS,
            key,
            sub_progress=sub_progress,
            message=message,
            extra_info=extra_info,
        )


class SshTrainingBackend:
    """Provision an SSH trainer container for one job, then train through it."""

    def __init__(
        self,
        job_id: UUID,
        server: RemoteServer,
        *,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ) -> None:
        self._job_id = job_id
        self._server = server
        self._protocol_version = protocol_version
        self._settings = get_settings()

    async def train(self, context: TrainingContext) -> None:
        """Provision (or reattach), then train through the provisioned trainer.

        When ``context.remote_job_id`` is set, a previous run already
        provisioned a container for this job; reattach to it instead of
        provisioning a fresh one.
        """
        if context.remote_job_id:
            await self._reattach_and_train(context)
            return
        await self._provision_and_train(context)

    async def _provision_and_train(self, context: TrainingContext) -> None:
        if context.snapshot is None:
            raise ValueError("SSH training requires a dataset snapshot")

        snapshot_path = Path(context.snapshot.path)
        snapshot_size_bytes = await asyncio.to_thread(_directory_size_bytes, snapshot_path)
        current_phase = PhaseKey.CONNECT

        def _report(
            key: PhaseKey,
            *,
            state: PhaseState = PhaseState.ACTIVE,
            sub_progress: float | None = 0.0,
            message: str | None = None,
            extra_info: dict | None = None,
        ) -> None:
            nonlocal current_phase
            current_phase = key
            report_phase(
                context.progress,
                SSH_PHASE_WINDOWS,
                key,
                state=state,
                sub_progress=sub_progress,
                message=message,
                extra_info=extra_info,
            )

        async def _on_gpu_wait(elapsed_s: float) -> None:
            if context.should_stop():
                raise TrainingCanceledError("Training canceled while waiting for a busy remote GPU")
            _report(
                PhaseKey.TRAINER_START,
                state=PhaseState.WAITING,
                sub_progress=None,
                message=f"Waiting for GPU to free up on remote server '{self._server.name}'",
                extra_info={"elapsed_s": round(elapsed_s)},
            )

        def _on_phase(key: PhaseKey) -> None:
            # Docker pull output has no stable percentage; pin it to the
            # window start and let the UI show a spinner instead.
            _report(key, sub_progress=None if key is PhaseKey.IMAGE_PULL else 0.0)

        _report(PhaseKey.CONNECT, message=f"Provisioning trainer on remote server '{self._server.name}'")
        try:
            async with get_async_db_session_ctx() as session:
                provisioning_service = SshProvisioningService(JobProvisioningRepository(session), self._settings)
                trainer = await provisioning_service.provision(
                    self._job_id,
                    self._server,
                    protocol_version=self._protocol_version,
                    snapshot_size_bytes=snapshot_size_bytes,
                    on_gpu_wait=_on_gpu_wait,
                    on_phase=_on_phase,
                )
        except BaseException:
            _report(current_phase, state=PhaseState.FAILED, sub_progress=None)
            raise

        await self._run_and_teardown(context, trainer)

    async def _reattach_and_train(self, context: TrainingContext) -> None:
        async with get_async_db_session_ctx() as session:
            repository = JobProvisioningRepository(session)
            job_provisioning = await repository.get_by_job_id(self._job_id)
            if job_provisioning is None:
                raise RemoteTrainingError(f"No provisioning record found for job {self._job_id} to reattach to")
            provisioning_service = SshProvisioningService(repository, self._settings)
            trainer = await provisioning_service.reattach(job_provisioning, self._server)

        if trainer is None:
            raise RemoteTrainingError(
                f"Trainer container for job {self._job_id} on remote server '{self._server.name}' is no longer "
                "running; it cannot be reattached to."
            )

        await self._run_and_teardown(context, trainer)

    async def _run_and_teardown(self, context: TrainingContext, trainer: ProvisionedTrainer) -> None:
        """Delegate training over the tunnel, then always tear down.

        The one exception is `TrainingSuspendedError`: the studio is shutting
        down, and the whole point of that path is to leave the container (and
        therefore the trainer job) running so a restart can reattach.

        `context.progress` is swapped for a phase-tagging wrapper for the
        duration of the delegated call, then restored, so `RemoteTrainingBackend`
        never has to know an SSH phase table exists while its own upload/train/
        download progress still lands in the right slice of the SSH stepper.
        """
        remote_backend = RemoteTrainingBackend(
            trainer.base_url,
            trainer_name=self._server.name,
            device=TrainingDevice(type=self._server.device_type, index=_INDEXED_DEVICE_INDEX),
        )
        original_progress = context.progress
        context.progress = _PhaseTaggingProgress(original_progress)
        try:
            try:
                await remote_backend.train(context)
            except TrainingSuspendedError:
                logger.info(
                    "Studio shutting down; leaving SSH-provisioned trainer container '{}' running for reattach",
                    trainer.container_name,
                )
                raise
            except BaseException:
                await self._teardown(trainer)
                raise
            else:
                await self._teardown(trainer)
        finally:
            context.progress = original_progress

    async def _teardown(self, trainer: ProvisionedTrainer) -> None:
        """Tear down the tunnel/container and drop the provisioning record."""
        await trainer.teardown()
        async with get_async_db_session_ctx() as session:
            await JobProvisioningRepository(session).delete_by_job_id(self._job_id)


__all__ = ["SshTrainingBackend"]
