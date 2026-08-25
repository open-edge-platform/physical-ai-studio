from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from exceptions import (
    RemoteResumeUnsupportedError,
    RemoteServerAliasNotFoundError,
    RemoteServerNotReadyError,
    ResourceNotFoundError,
    ResourceType,
)
from schemas.hardware import DeviceType
from schemas.job import RemoteTrainJobPayload, SshTrainJobPayload
from schemas.remote_server import RemoteServer, RemoteServerCheckStatus, ResolvedSshHost
from schemas.remote_trainer import RemoteTrainer
from services.job_service import JobService

MODULE = "services.job_service"
SSH_MODULE = "services.training_targets.ssh"


def _session_context() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.mark.anyio
async def test_submit_remote_job_pins_configured_url_and_ignores_client_url() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    remote_trainer_id = uuid4()
    configured_trainer = RemoteTrainer(
        id=remote_trainer_id,
        name="trainer",
        url="https://configured-trainer.test",
    )
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_trainer_id=remote_trainer_id,
        remote_trainer_url="https://client-supplied.test",
    )

    remote_trainer_service = MagicMock()
    remote_trainer_service.get_remote_trainer = AsyncMock(return_value=configured_trainer)

    with patch(f"{MODULE}.JobRepository", return_value=repository):
        job = await JobService(session, remote_trainer_service).submit_train_job(payload)

    assert str(job.payload.remote_trainer_url) == "https://configured-trainer.test/"
    assert job.payload.remote_trainer_id == remote_trainer_id
    # Pinned from the configured trainer's record, for display in job logs.
    assert job.payload.remote_trainer_name == "trainer"
    repository.save.assert_awaited_once()


@pytest.mark.anyio
async def test_submit_remote_job_rejects_continuing_from_an_existing_model() -> None:
    """The trainer protocol has no way to upload a base checkpoint.

    Without this the job would be accepted and silently trained from scratch,
    so the rejection happens at submission rather than minutes later.
    """
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_trainer_id=uuid4(),
        base_model_id=uuid4(),
    )

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        pytest.raises(RemoteResumeUnsupportedError),
    ):
        await JobService(session, MagicMock()).submit_train_job(payload)

    repository.save.assert_not_awaited()


def _remote_server(*, last_check_status: RemoteServerCheckStatus = "healthy") -> RemoteServer:
    return RemoteServer(
        id=uuid4(),
        name="gpu-box",
        ssh_host_alias="gpu-box",
        device_type=DeviceType.CUDA,
        last_check_status=last_check_status,
    )


@pytest.mark.anyio
async def test_submit_ssh_job_accepts_a_healthy_server() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    remote_server = _remote_server()
    payload = SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=remote_server.id,
    )

    remote_server_service = MagicMock()
    remote_server_service.get_remote_server = AsyncMock(return_value=remote_server)

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        patch(
            f"{SSH_MODULE}.resolve_alias",
            return_value=ResolvedSshHost(alias=remote_server.ssh_host_alias, hostname="gpu-box.lan", found=True),
        ),
    ):
        job = await JobService(session, remote_server_service=remote_server_service).submit_train_job(payload)

    assert job.payload.remote_server_id == remote_server.id
    repository.save.assert_awaited_once()


@pytest.mark.anyio
async def test_submit_ssh_job_rejects_an_unhealthy_server() -> None:
    """A server that has not passed its last preflight cannot be selected for a job."""
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    remote_server = _remote_server(last_check_status="unreachable")
    payload = SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=remote_server.id,
    )

    remote_server_service = MagicMock()
    remote_server_service.get_remote_server = AsyncMock(return_value=remote_server)

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        pytest.raises(RemoteServerNotReadyError),
    ):
        await JobService(session, remote_server_service=remote_server_service).submit_train_job(payload)

    repository.save.assert_not_awaited()


@pytest.mark.anyio
async def test_submit_ssh_job_rejects_a_renamed_or_removed_alias() -> None:
    """A healthy server whose Host entry has since disappeared still fails closed."""
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    remote_server = _remote_server()
    payload = SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=remote_server.id,
    )

    remote_server_service = MagicMock()
    remote_server_service.get_remote_server = AsyncMock(return_value=remote_server)

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        patch(
            f"{SSH_MODULE}.resolve_alias",
            return_value=ResolvedSshHost(alias=remote_server.ssh_host_alias, found=False),
        ),
        pytest.raises(RemoteServerAliasNotFoundError),
    ):
        await JobService(session, remote_server_service=remote_server_service).submit_train_job(payload)

    repository.save.assert_not_awaited()


@pytest.mark.anyio
async def test_submit_ssh_job_rejects_an_unknown_server() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    payload = SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=uuid4(),
    )

    remote_server_service = MagicMock()
    remote_server_service.get_remote_server = AsyncMock(
        side_effect=ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(payload.remote_server_id))
    )

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        pytest.raises(ResourceNotFoundError),
    ):
        await JobService(session, remote_server_service=remote_server_service).submit_train_job(payload)

    repository.save.assert_not_awaited()


@pytest.mark.anyio
async def test_submit_ssh_job_rejects_continuing_from_an_existing_model() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.is_job_duplicate = AsyncMock(return_value=False)
    repository.save = AsyncMock(side_effect=lambda job: job)
    payload = SshTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        policy="act",
        model_name="model",
        remote_server_id=uuid4(),
        base_model_id=uuid4(),
    )

    with (
        patch(f"{MODULE}.JobRepository", return_value=repository),
        pytest.raises(RemoteResumeUnsupportedError),
    ):
        await JobService(session, MagicMock()).submit_train_job(payload)

    repository.save.assert_not_awaited()
