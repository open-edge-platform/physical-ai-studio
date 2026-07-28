from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError
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
async def test_delete_missing_remote_trainer_raises_not_found() -> None:
    session = _session()
    repository = MagicMock()
    repository.get_by_id = AsyncMock(return_value=None)

    with patch(f"{MODULE}.RemoteTrainerRepository", return_value=repository), pytest.raises(ResourceNotFoundError):
        await RemoteTrainerService(session).delete_remote_trainer(uuid4())

    repository.delete_by_id.assert_not_called()
