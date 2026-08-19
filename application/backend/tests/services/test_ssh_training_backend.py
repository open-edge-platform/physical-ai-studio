# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SSH-provisioned training backend.

`SshProvisioningService` and `RemoteTrainingBackend` are faked: this proves the
*orchestration* SshTrainingBackend adds on top of them (provision-vs-reattach
dispatch, device injection, and teardown-on-every-path-but-suspend), not their
own already-tested internals.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from schemas.hardware import DeviceType
from schemas.job import TrainingDevice
from schemas.remote_server import RemoteServer
from services.training_backends.base import TrainingCanceledError, TrainingSuspendedError
from services.training_backends.ssh import SshTrainingBackend

MODULE = "services.training_backends.ssh"


def _server() -> RemoteServer:
    return RemoteServer(id=uuid4(), name="Lab GPU box", ssh_host_alias="gpu-box", device_type=DeviceType.CUDA)


def _context(*, remote_job_id=None, should_stop=False) -> MagicMock:
    context = MagicMock()
    context.remote_job_id = remote_job_id
    context.snapshot = None
    context.should_stop = MagicMock(return_value=should_stop)
    context.on_remote_job_id = AsyncMock()
    return context


@asynccontextmanager
async def _session_scope():
    yield MagicMock()


def _patched_provisioning_service(provisioned_trainer):
    service = MagicMock()
    service.provision = AsyncMock(return_value=provisioned_trainer)
    service.reattach = AsyncMock(return_value=provisioned_trainer)
    return service


async def test_train_provisions_and_tears_down_on_success() -> None:
    server = _server()
    trainer = MagicMock()
    trainer.base_url = "http://127.0.0.1:54321"
    trainer.container_name = "physicalai-trainer-abc"
    trainer.teardown = AsyncMock()

    remote_backend = MagicMock()
    remote_backend.train = AsyncMock()

    service = _patched_provisioning_service(trainer)
    repo = MagicMock()
    repo.delete_by_job_id = AsyncMock()

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
        patch(f"{MODULE}.RemoteTrainingBackend", return_value=remote_backend) as MockRemote,
    ):
        backend = SshTrainingBackend(uuid4(), server)
        context = _context()
        await backend.train(context)

    service.provision.assert_awaited_once()
    remote_backend.train.assert_awaited_once_with(context)
    trainer.teardown.assert_awaited_once()
    repo.delete_by_job_id.assert_awaited_once()
    # The server's own configured device is injected, not probed live.
    _, kwargs = MockRemote.call_args
    assert kwargs["device"] == TrainingDevice(type=DeviceType.CUDA, index=0)


async def test_train_tears_down_on_failure() -> None:
    server = _server()
    trainer = MagicMock()
    trainer.base_url = "http://127.0.0.1:54321"
    trainer.container_name = "physicalai-trainer-abc"
    trainer.teardown = AsyncMock()

    remote_backend = MagicMock()
    remote_backend.train = AsyncMock(side_effect=RuntimeError("boom"))

    service = _patched_provisioning_service(trainer)
    repo = MagicMock()
    repo.delete_by_job_id = AsyncMock()

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
        patch(f"{MODULE}.RemoteTrainingBackend", return_value=remote_backend),
    ):
        backend = SshTrainingBackend(uuid4(), server)
        context = _context()
        try:
            await backend.train(context)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError to propagate")

    trainer.teardown.assert_awaited_once()
    repo.delete_by_job_id.assert_awaited_once()


async def test_train_leaves_container_running_on_suspend() -> None:
    """A TrainingSuspendedError must not tear anything down - it's for reattach."""
    server = _server()
    trainer = MagicMock()
    trainer.base_url = "http://127.0.0.1:54321"
    trainer.container_name = "physicalai-trainer-abc"
    trainer.teardown = AsyncMock()

    remote_backend = MagicMock()
    remote_backend.train = AsyncMock(side_effect=TrainingSuspendedError("shutting down"))

    service = _patched_provisioning_service(trainer)
    repo = MagicMock()
    repo.delete_by_job_id = AsyncMock()

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
        patch(f"{MODULE}.RemoteTrainingBackend", return_value=remote_backend),
    ):
        backend = SshTrainingBackend(uuid4(), server)
        context = _context()
        try:
            await backend.train(context)
        except TrainingSuspendedError:
            pass
        else:
            raise AssertionError("expected TrainingSuspendedError to propagate")

    trainer.teardown.assert_not_called()
    repo.delete_by_job_id.assert_not_called()


async def test_reattach_uses_persisted_provisioning_row() -> None:
    server = _server()
    job_id = uuid4()
    job_provisioning = MagicMock()

    trainer = MagicMock()
    trainer.base_url = "http://127.0.0.1:54321"
    trainer.container_name = "physicalai-trainer-abc"
    trainer.teardown = AsyncMock()

    remote_backend = MagicMock()
    remote_backend.train = AsyncMock()

    service = _patched_provisioning_service(trainer)
    repo = MagicMock()
    repo.get_by_job_id = AsyncMock(return_value=job_provisioning)
    repo.delete_by_job_id = AsyncMock()

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
        patch(f"{MODULE}.RemoteTrainingBackend", return_value=remote_backend),
    ):
        backend = SshTrainingBackend(job_id, server)
        context = _context(remote_job_id=uuid4())
        await backend.train(context)

    service.reattach.assert_awaited_once_with(job_provisioning, server)
    remote_backend.train.assert_awaited_once_with(context)
    trainer.teardown.assert_awaited_once()


async def test_reattach_raises_when_container_gone() -> None:
    server = _server()
    job_id = uuid4()
    job_provisioning = MagicMock()

    service = _patched_provisioning_service(None)
    service.reattach = AsyncMock(return_value=None)
    repo = MagicMock()
    repo.get_by_job_id = AsyncMock(return_value=job_provisioning)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
    ):
        backend = SshTrainingBackend(job_id, server)
        context = _context(remote_job_id=uuid4())
        try:
            await backend.train(context)
        except Exception as error:
            assert "no longer running" in str(error)
        else:
            raise AssertionError("expected an error when the container is gone")


async def test_gpu_wait_callback_raises_when_canceled() -> None:
    """A cancellation requested while waiting for a busy GPU aborts the wait."""
    server = _server()
    service = MagicMock()

    async def _provision(*_args, **kwargs):
        await kwargs["on_gpu_wait"](5.0)
        raise AssertionError("on_gpu_wait should have raised before provision continued")

    service.provision = AsyncMock(side_effect=_provision)
    repo = MagicMock()

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", _session_scope),
        patch(f"{MODULE}.JobProvisioningRepository", return_value=repo),
        patch(f"{MODULE}.SshProvisioningService", return_value=service),
    ):
        backend = SshTrainingBackend(uuid4(), server)
        context = _context(should_stop=True)
        try:
            await backend.train(context)
        except TrainingCanceledError:
            pass
        else:
            raise AssertionError("expected TrainingCanceledError")
