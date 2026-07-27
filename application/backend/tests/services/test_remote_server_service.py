from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from exceptions import ResourceAlreadyExistsError, ResourceInUseError, ResourceNotFoundError
from schemas.hardware import DeviceType
from schemas.remote_server import (
    LastCheckSummary,
    RemoteServerCreate,
    RemoteServerInternal,
    RemoteServerUpdate,
    SSHAuthType,
)
from services import RemoteServerService

MODULE = "services.remote_server_service"


def _session_context() -> AsyncMock:
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _record(**overrides) -> RemoteServerInternal:
    defaults = {
        "id": uuid4(),
        "name": "gpu-box",
        "host": "10.0.0.5",
        "port": 22,
        "username": "trainer",
        "auth_type": SSHAuthType.PASSWORD,
        "device_type": DeviceType.CUDA,
        "last_check": LastCheckSummary(),
        "ssh_secret_encrypted": "ciphertext",
        "ssh_key_passphrase_encrypted": None,
        "host_key": None,
    }
    defaults.update(overrides)
    return RemoteServerInternal(**defaults)


def _create_config(**overrides) -> RemoteServerCreate:
    defaults = {
        "name": "gpu-box",
        "host": "10.0.0.5",
        "username": "trainer",
        "auth_type": SSHAuthType.PASSWORD,
        "device_type": DeviceType.CUDA,
        "ssh_secret": "hunter2",
    }
    defaults.update(overrides)
    return RemoteServerCreate(**defaults)


@pytest.mark.anyio
async def test_list_remote_servers_returns_sanitized_public_view() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[_record()])

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
    ):
        result = await RemoteServerService.list_remote_servers()

    assert len(result) == 1
    assert not hasattr(result[0], "ssh_secret_encrypted")


@pytest.mark.anyio
async def test_get_remote_server_raises_not_found() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        pytest.raises(ResourceNotFoundError),
    ):
        await RemoteServerService.get_remote_server(uuid4())


@pytest.mark.anyio
async def test_create_remote_server_encrypts_secret_before_persisting() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=lambda item: item)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        patch(f"{MODULE}.encrypt_secret", return_value="encrypted-value") as encrypt_mock,
    ):
        result = await RemoteServerService.create_remote_server(_create_config())

    encrypt_mock.assert_called_once_with("hunter2")
    await_args = repository.save.await_args
    assert await_args is not None
    saved_record = await_args.args[0]
    assert saved_record.ssh_secret_encrypted == "encrypted-value"
    assert not hasattr(result, "ssh_secret_encrypted")


@pytest.mark.anyio
async def test_create_remote_server_encrypts_passphrase_when_present() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=lambda item: item)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        patch(f"{MODULE}.encrypt_secret", side_effect=lambda value: f"enc:{value}") as encrypt_mock,
    ):
        await RemoteServerService.create_remote_server(
            _create_config(auth_type=SSHAuthType.KEY, ssh_secret="key-data", ssh_key_passphrase="unlock")
        )

    encrypt_mock.assert_any_call("key-data")
    encrypt_mock.assert_any_call("unlock")
    await_args = repository.save.await_args
    assert await_args is not None
    saved_record = await_args.args[0]
    assert saved_record.ssh_secret_encrypted == "enc:key-data"
    assert saved_record.ssh_key_passphrase_encrypted == "enc:unlock"


@pytest.mark.anyio
async def test_create_duplicate_remote_server_returns_conflict() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=IntegrityError("insert", {}, Exception("duplicate")))

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        patch(f"{MODULE}.encrypt_secret", return_value="encrypted-value"),
        pytest.raises(ResourceAlreadyExistsError) as error,
    ):
        await RemoteServerService.create_remote_server(_create_config())

    assert error.value.http_status == 409
    session.rollback.assert_awaited_once_with()


@pytest.mark.anyio
async def test_update_rotates_encrypted_secret_only_when_provided() -> None:
    session = _session_context()
    record = _record()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=record)
    repository.update = AsyncMock(side_effect=lambda item, partial: record)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        patch(f"{MODULE}.encrypt_secret", return_value="rotated-ciphertext") as encrypt_mock,
    ):
        await RemoteServerService.update_remote_server(record.id, RemoteServerUpdate(ssh_secret="new-secret"))

    encrypt_mock.assert_called_once_with("new-secret")
    await_args = repository.update.await_args
    assert await_args is not None
    partial_update = await_args.args[1]
    assert partial_update["ssh_secret_encrypted"] == "rotated-ciphertext"
    assert "ssh_secret" not in partial_update


@pytest.mark.anyio
async def test_update_without_secret_fields_does_not_touch_encryption() -> None:
    session = _session_context()
    record = _record()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=record)
    repository.update = AsyncMock(return_value=record)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        patch(f"{MODULE}.encrypt_secret") as encrypt_mock,
    ):
        await RemoteServerService.update_remote_server(record.id, RemoteServerUpdate(name="renamed"))

    encrypt_mock.assert_not_called()
    await_args = repository.update.await_args
    assert await_args is not None
    partial_update = await_args.args[1]
    assert partial_update == {"name": "renamed"}


@pytest.mark.anyio
async def test_delete_missing_remote_server_raises_not_found() -> None:
    session = _session_context()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        pytest.raises(ResourceNotFoundError),
    ):
        await RemoteServerService.delete_remote_server(uuid4())

    repository.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_delete_remote_server_in_use_raises_resource_in_use() -> None:
    session = _session_context()
    record = _record()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=record)
    repository.delete_by_id = AsyncMock(side_effect=IntegrityError("delete", {}, Exception("fk constraint")))

    with (
        patch(f"{MODULE}.get_async_db_session_ctx", return_value=session),
        patch(f"{MODULE}.RemoteServerRepository", return_value=repository),
        pytest.raises(ResourceInUseError) as error,
    ):
        await RemoteServerService.delete_remote_server(record.id)

    assert error.value.http_status == 409
    session.rollback.assert_awaited_once_with()


def test_decrypt_helpers_delegate_to_secret_encryption() -> None:
    record = _record(ssh_secret_encrypted="enc-secret", ssh_key_passphrase_encrypted="enc-pass")

    with patch(f"{MODULE}.decrypt_secret", side_effect=lambda value: f"plain:{value}") as decrypt_mock:
        assert RemoteServerService.decrypt_ssh_secret(record) == "plain:enc-secret"
        assert RemoteServerService.decrypt_ssh_key_passphrase(record) == "plain:enc-pass"

    assert decrypt_mock.call_count == 2


def test_decrypt_passphrase_returns_none_when_not_configured() -> None:
    record = _record(ssh_key_passphrase_encrypted=None)

    assert RemoteServerService.decrypt_ssh_key_passphrase(record) is None
