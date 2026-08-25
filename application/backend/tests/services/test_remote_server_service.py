# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError
from schemas.hardware import DeviceType
from schemas.job_provisioning import JobProvisioning
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightCheck, PreflightResult
from services import RemoteServerService
from services.remote_server_service import reset_status_cache
from settings import Settings

MODULE = "services.remote_server_service"
CHECKED_AT = datetime.now(UTC)


@pytest.fixture(autouse=True)
def _clear_status_cache():
    """Every test starts and ends with a clean per-server status cache/in-flight table."""
    reset_status_cache()
    yield
    reset_status_cache()


def _job_provisioning(remote_server_id, job_id=None) -> JobProvisioning:
    return JobProvisioning(
        job_id=job_id or uuid4(),
        remote_server_id=remote_server_id,
        ssh_host_alias="gpu-box",
        container_name="physicalai-trainer-abc",
    )


class _FakeJob:
    """Minimal stand-in for `schemas.Job`; only `extra_info` is read by the service."""

    def __init__(self, extra_info: dict | None) -> None:
        self.extra_info = extra_info


def _tier1_result(*, passed: bool = True) -> PreflightResult:
    outcome = CheckOutcome.PASSED if passed else CheckOutcome.FAILED
    return PreflightResult(
        checks=[
            PreflightCheck(
                key=CheckKey.DOCKER_USABLE,
                tier=1,  # type: ignore[arg-type]
                outcome=outcome,
                blocking=True,
                checked_at=CHECKED_AT,
                reason_code=None if passed else "docker_unavailable",
            )
        ],
        checked_at=CHECKED_AT,
        latency_ms=10,
    )


def _session() -> AsyncMock:
    return AsyncMock()


def _remote_server() -> RemoteServer:
    return RemoteServer(id=uuid4(), name="server", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)


def _check(outcome: CheckOutcome, *, blocking: bool = True, reason_code: str | None = None) -> PreflightCheck:
    return PreflightCheck(
        key=CheckKey.IMAGE_RESOLVED,
        tier=2,  # type: ignore[arg-type]
        outcome=outcome,
        blocking=blocking,
        checked_at=CHECKED_AT,
        reason_code=reason_code,
    )


@pytest.mark.anyio
async def test_list_remote_servers_uses_stable_repository_order() -> None:
    session = _session()
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[_remote_server()])

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        result = await RemoteServerService(session).list_remote_servers()

    assert result == [repository.list_ordered.return_value[0]]
    repository.list_ordered.assert_awaited_once_with()


@pytest.mark.anyio
async def test_get_remote_server_returns_match() -> None:
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        result = await RemoteServerService(session).get_remote_server(remote_server.id)

    assert result == remote_server


@pytest.mark.anyio
async def test_get_missing_remote_server_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteServerService(session).get_remote_server(uuid4())


@pytest.mark.anyio
async def test_create_remote_server_persists_via_repository() -> None:
    session = _session()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=lambda item: item)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        result = await RemoteServerService(session).create_remote_server(
            RemoteServerCreate(name="server", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)
        )

    assert result.name == "server"
    assert result.ssh_host_alias == "my-gpu-box"
    repository.save.assert_awaited_once()


@pytest.mark.anyio
async def test_create_duplicate_remote_server_returns_conflict() -> None:
    session = _session()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=IntegrityError("insert", {}, Exception("duplicate")))

    with (
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        pytest.raises(ResourceAlreadyExistsError) as error,
    ):
        await RemoteServerService(session).create_remote_server(
            RemoteServerCreate(name="server", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)
        )

    assert error.value.http_status == 409
    session.rollback.assert_awaited_once_with()


@pytest.mark.anyio
async def test_update_ignores_explicit_null_fields() -> None:
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)
    repository.update = AsyncMock(return_value=remote_server)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        await RemoteServerService(session).update_remote_server(remote_server.id, RemoteServerUpdate(name=None))

    repository.update.assert_awaited_once_with(remote_server, {})


@pytest.mark.anyio
async def test_update_missing_remote_server_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteServerService(session).update_remote_server(uuid4(), RemoteServerUpdate(name="new name"))

    repository.update.assert_not_called()


@pytest.mark.anyio
async def test_update_duplicate_remote_server_returns_conflict() -> None:
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)
    repository.update = AsyncMock(side_effect=IntegrityError("update", {}, Exception("duplicate")))

    with (
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        pytest.raises(ResourceAlreadyExistsError) as error,
    ):
        await RemoteServerService(session).update_remote_server(
            remote_server.id, RemoteServerUpdate(ssh_host_alias="other-box")
        )

    assert error.value.http_status == 409
    session.rollback.assert_awaited_once_with()


@pytest.mark.anyio
async def test_delete_remote_server_deletes_by_id() -> None:
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)
    repository.delete_by_id = AsyncMock()

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        await RemoteServerService(session).delete_remote_server(remote_server.id)

    repository.delete_by_id.assert_awaited_once_with(remote_server.id)


@pytest.mark.anyio
async def test_delete_missing_remote_server_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteServerService(session).delete_remote_server(uuid4())

    repository.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_record_check_result_persists_healthy_on_pass() -> None:
    """A passing Tier 2 result must move `last_check_status` to "healthy".

    Regression guard for the bug where `last_check_status` stayed "unknown"
    forever because nothing ever wrote it back to the DB.
    """
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)
    repository.update = AsyncMock(return_value=remote_server)
    result = PreflightResult(checks=[_check(CheckOutcome.PASSED)], checked_at=CHECKED_AT, latency_ms=1200)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        await RemoteServerService(session).record_check_result(remote_server.id, result)

    repository.update.assert_awaited_once_with(
        remote_server,
        {
            "last_check_status": "healthy",
            "last_check_at": result.checked_at,
            "last_check_latency_ms": 1200,
            "last_check_reason_code": None,
            "last_check_checks": [check.model_dump(mode="json") for check in result.checks],
        },
    )


@pytest.mark.anyio
async def test_record_check_result_persists_degraded_on_blocking_failure() -> None:
    session = _session()
    remote_server = _remote_server()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_server)
    repository.update = AsyncMock(return_value=remote_server)
    result = PreflightResult(
        checks=[_check(CheckOutcome.FAILED, blocking=True, reason_code="image_pull_failed")],
        checked_at=CHECKED_AT,
        latency_ms=500,
    )

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository):
        await RemoteServerService(session).record_check_result(remote_server.id, result)

    repository.update.assert_awaited_once_with(
        remote_server,
        {
            "last_check_status": "degraded",
            "last_check_at": result.checked_at,
            "last_check_latency_ms": 500,
            "last_check_reason_code": "image_pull_failed",
            "last_check_checks": [check.model_dump(mode="json") for check in result.checks],
        },
    )


@pytest.mark.anyio
async def test_record_check_result_missing_remote_server_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)
    result = PreflightResult(checks=[_check(CheckOutcome.PASSED)], checked_at=CHECKED_AT, latency_ms=100)

    with patch(f"{MODULE}.RemoteServerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteServerService(session).record_check_result(uuid4(), result)

    repository.update.assert_not_called()


def _patched_service(
    remote_server: RemoteServer,
    *,
    active: JobProvisioning | None = None,
    job: object | None = None,
    tier1_result: PreflightResult | None = None,
) -> tuple[RemoteServerService, AsyncMock, MagicMock, MagicMock]:
    """Build a `RemoteServerService` whose repositories/preflight are mocked.

    Returns the service plus the `run_tier1_preflight` mock, so a test can
    assert how many times the SSH probe actually ran.
    """
    server_repo = MagicMock()
    server_repo.get_by_id = AsyncMock(return_value=remote_server)

    provisioning_repo = MagicMock()
    provisioning_repo.get_active_for_server = AsyncMock(return_value=active)

    job_repo = MagicMock()
    job_repo.get_by_id = AsyncMock(return_value=job)

    run_tier1 = AsyncMock(return_value=tier1_result or _tier1_result())

    service = RemoteServerService(_session())
    service.repo = server_repo
    return service, run_tier1, provisioning_repo, job_repo


@pytest.mark.anyio
async def test_get_status_reports_not_in_use_when_no_active_provisioning() -> None:
    remote_server = _remote_server()
    service, run_tier1, provisioning_repo, job_repo = _patched_service(remote_server)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=5.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", run_tier1),
    ):
        status = await service.get_status(remote_server.id, settings)

    assert status.in_use_by_job_id is None
    assert status.waiting_for_gpu is False
    job_repo.get_by_id.assert_not_called()


@pytest.mark.anyio
async def test_get_status_reports_in_use_job_and_waiting_for_gpu_phase() -> None:
    remote_server = _remote_server()
    active = _job_provisioning(remote_server.id)
    waiting_job = _FakeJob(extra_info={"phase": {"key": "trainer_start", "state": "waiting"}})
    service, run_tier1, provisioning_repo, job_repo = _patched_service(remote_server, active=active, job=waiting_job)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=5.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", run_tier1),
    ):
        status = await service.get_status(remote_server.id, settings)

    assert status.in_use_by_job_id == active.job_id
    assert status.waiting_for_gpu is True


@pytest.mark.anyio
async def test_get_status_does_not_report_waiting_for_gpu_when_phase_is_active() -> None:
    remote_server = _remote_server()
    active = _job_provisioning(remote_server.id)
    training_job = _FakeJob(extra_info={"phase": {"key": "train", "state": "active"}})
    service, run_tier1, provisioning_repo, job_repo = _patched_service(remote_server, active=active, job=training_job)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=5.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", run_tier1),
    ):
        status = await service.get_status(remote_server.id, settings)

    assert status.in_use_by_job_id == active.job_id
    assert status.waiting_for_gpu is False


@pytest.mark.anyio
async def test_get_status_throttles_repeated_calls_within_window() -> None:
    """A second call within `ssh_preflight_throttle_s` must reuse the cached
    Tier 1 result instead of dialing out again - the whole point of the
    per-server throttle the status endpoint shares with polling.
    """
    remote_server = _remote_server()
    service, run_tier1, provisioning_repo, job_repo = _patched_service(remote_server)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=60.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", run_tier1),
    ):
        first = await service.get_status(remote_server.id, settings)
        second = await service.get_status(remote_server.id, settings)

    assert first == second
    run_tier1.assert_awaited_once()


@pytest.mark.anyio
async def test_get_status_coalesces_concurrent_calls_into_one_probe() -> None:
    """Two concurrent pollers for the same server must share one SSH dial."""
    remote_server = _remote_server()
    service, _, provisioning_repo, job_repo = _patched_service(remote_server)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=60.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    call_count = 0

    async def _slow_tier1(_server: RemoteServer) -> PreflightResult:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return _tier1_result()

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", _slow_tier1),
    ):
        first, second = await asyncio.gather(
            service.get_status(remote_server.id, settings),
            service.get_status(remote_server.id, settings),
        )

    assert first == second
    assert call_count == 1


@pytest.mark.anyio
async def test_get_status_re_probes_after_throttle_window_expires() -> None:
    remote_server = _remote_server()
    service, run_tier1, provisioning_repo, job_repo = _patched_service(remote_server)
    settings = Settings(SSH_PREFLIGHT_THROTTLE_S=0.0, SSH_PREFLIGHT_TIMEOUT_S=30.0)

    with (
        patch(f"{MODULE}.JobProvisioningRepository", return_value=provisioning_repo),
        patch(f"{MODULE}.JobRepository", return_value=job_repo),
        patch(f"{MODULE}.run_tier1_preflight", run_tier1),
    ):
        await service.get_status(remote_server.id, settings)
        await service.get_status(remote_server.id, settings)

    assert run_tier1.await_count == 2
