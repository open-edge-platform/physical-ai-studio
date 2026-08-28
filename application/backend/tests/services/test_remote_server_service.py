# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate
from schemas.ssh_preflight import CheckKey, CheckOutcome, PreflightCheck, PreflightResult
from services import RemoteServerService

MODULE = "services.remote_server_service"
CHECKED_AT = datetime.now(UTC)


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
