# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError
from schemas.hardware import DeviceType
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate
from services import RemoteServerService

MODULE = "services.remote_server_service"


def _session() -> AsyncMock:
    return AsyncMock()


def _remote_server() -> RemoteServer:
    return RemoteServer(id=uuid4(), name="server", ssh_host_alias="my-gpu-box", device_type=DeviceType.CUDA)


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
