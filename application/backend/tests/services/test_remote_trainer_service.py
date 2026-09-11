from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from core.security.ssh_network_exposure import SshFeatureAvailability
from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, SshFeatureDisabledError
from schemas.remote_trainer import RemoteTrainer, RemoteTrainerCreate, RemoteTrainerUpdate
from services import RemoteTrainerService

MODULE = "services.remote_trainer_service"


def _session() -> AsyncMock:
    return AsyncMock()


def _remote_trainer() -> RemoteTrainer:
    return RemoteTrainer(id=uuid4(), name="trainer", url="https://trainer.test")


def test_remote_trainer_name_is_trimmed() -> None:
    config = RemoteTrainerCreate(name="  trainer  ", url="https://trainer.test")

    assert config.name == "trainer"


def test_remote_trainer_rejects_whitespace_only_name() -> None:
    with pytest.raises(ValidationError):
        RemoteTrainerCreate(name="   ", url="https://trainer.test")


@pytest.mark.anyio
async def test_list_remote_trainers_uses_stable_repository_order() -> None:
    session = _session()
    repository = MagicMock()
    repository.list_ordered = AsyncMock(return_value=[_remote_trainer()])

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository):
        result = await RemoteTrainerService(session).list_remote_trainers()

    assert result == [repository.list_ordered.return_value[0]]
    repository.list_ordered.assert_awaited_once_with()


@pytest.mark.anyio
async def test_create_duplicate_remote_trainer_returns_conflict() -> None:
    session = _session()
    repository = MagicMock()
    repository.save = AsyncMock(side_effect=IntegrityError("insert", {}, Exception("duplicate")))

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        pytest.raises(ResourceAlreadyExistsError) as error,
    ):
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="trainer", url="https://trainer.test")
        )

    assert error.value.http_status == 409
    session.rollback.assert_awaited_once_with()


@pytest.mark.anyio
async def test_update_ignores_explicit_null_fields() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.update = AsyncMock(return_value=remote_trainer)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository):
        await RemoteTrainerService(session).update_remote_trainer(remote_trainer.id, RemoteTrainerUpdate(name=None))

    repository.update.assert_awaited_once_with(remote_trainer, {})


@pytest.mark.anyio
async def test_update_clears_explicit_null_tunnel_fields() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.update = AsyncMock(return_value=remote_trainer)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository):
        await RemoteTrainerService(session).update_remote_trainer(
            remote_trainer.id,
            RemoteTrainerUpdate(ssh_host_alias=None, ssh_remote_port=None, ssh_local_port=None),
        )

    repository.update.assert_awaited_once_with(
        remote_trainer, {"ssh_host_alias": None, "ssh_remote_port": None, "ssh_local_port": None}
    )


@pytest.mark.anyio
async def test_delete_missing_remote_trainer_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteTrainerService(session).delete_remote_trainer(uuid4())

    repository.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_create_rejects_ssh_tunnel_config_when_feature_inactive() -> None:
    session = _session()
    repository = MagicMock()

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(
            f"{MODULE}.get_ssh_feature_availability",
            return_value=SshFeatureAvailability(network_exposed=True),
        ),
        pytest.raises(SshFeatureDisabledError),
    ):
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(
                name="trainer", url="http://127.0.0.1:8001", ssh_host_alias="training-box", ssh_local_port=8001
            )
        )

    repository.save.assert_not_called()


@pytest.mark.anyio
async def test_create_syncs_the_tunnel_manager_on_success() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.save = AsyncMock(return_value=remote_trainer)

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
    ):
        tunnel_manager.sync_tunnel = AsyncMock()
        await RemoteTrainerService(session).create_remote_trainer(
            RemoteTrainerCreate(name="trainer", url="https://trainer.test")
        )

    tunnel_manager.sync_tunnel.assert_awaited_once_with(remote_trainer)


@pytest.mark.anyio
async def test_delete_stops_the_tunnel_manager() -> None:
    session = _session()
    remote_trainer = _remote_trainer()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=remote_trainer)
    repository.delete_by_id = AsyncMock()

    with (
        patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository),
        patch(f"{MODULE}.remote_trainer_tunnel_manager") as tunnel_manager,
    ):
        tunnel_manager.stop_tunnel = AsyncMock()
        await RemoteTrainerService(session).delete_remote_trainer(remote_trainer.id)

    tunnel_manager.stop_tunnel.assert_awaited_once_with(remote_trainer.id)
