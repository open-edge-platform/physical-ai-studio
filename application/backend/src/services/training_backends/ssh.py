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
from services.training_backends.remote import RemoteTrainingBackend, RemoteTrainingError
from settings import get_settings

if TYPE_CHECKING:
    from uuid import UUID

    from schemas.remote_server import RemoteServer
    from services.training_backends.base import TrainingContext

# Trainer containers only ever run on cuda/xpu servers (see
# `schemas.remote_server.SSH_SERVER_DEVICE_TYPES`), so the accelerator always
# needs an index.
_INDEXED_DEVICE_INDEX: Final = 0


def _directory_size_bytes(path: Path) -> int:
    """Sum the size of every regular file under ``path`` (blocking)."""
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


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
        snapshot_size_bytes = 0
        if context.snapshot is not None:
            snapshot_size_bytes = await asyncio.to_thread(_directory_size_bytes, Path(context.snapshot.path))

        async def _on_gpu_wait(elapsed_s: float) -> None:
            if context.should_stop():
                raise TrainingCanceledError("Training canceled while waiting for a busy remote GPU")
            context.progress(
                0,
                message=f"Waiting for GPU to free up on remote server '{self._server.name}'",
                extra_info={"phase": {"key": "waiting", "elapsed_s": round(elapsed_s)}},
            )

        context.progress(0, message=f"Provisioning trainer on remote server '{self._server.name}'")
        async with get_async_db_session_ctx() as session:
            provisioning_service = SshProvisioningService(JobProvisioningRepository(session), self._settings)
            trainer = await provisioning_service.provision(
                self._job_id,
                self._server,
                protocol_version=self._protocol_version,
                snapshot_size_bytes=snapshot_size_bytes,
                on_gpu_wait=_on_gpu_wait,
            )

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
        """
        remote_backend = RemoteTrainingBackend(
            trainer.base_url,
            trainer_name=self._server.name,
            device=TrainingDevice(type=self._server.device_type, index=_INDEXED_DEVICE_INDEX),
        )
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

    async def _teardown(self, trainer: ProvisionedTrainer) -> None:
        """Tear down the tunnel/container and drop the provisioning record."""
        await trainer.teardown()
        async with get_async_db_session_ctx() as session:
            await JobProvisioningRepository(session).delete_by_job_id(self._job_id)


__all__ = ["SshTrainingBackend"]
