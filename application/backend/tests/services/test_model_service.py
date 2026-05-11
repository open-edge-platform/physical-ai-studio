"""Unit tests for ModelService focusing on the delete_model behaviour."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from schemas.model import Model
from services.model_service import ModelService


def _make_model(snapshot_id=None) -> Model:
    return Model.model_validate(
        {
            "id": str(uuid4()),
            "name": "test-model",
            "policy": "act",
            "path": "/tmp/test-model",
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "snapshot_id": str(snapshot_id) if snapshot_id else None,
            "properties": {},
        }
    )


@pytest.mark.anyio
async def test_delete_model_deletes_snapshot_when_snapshot_id_set() -> None:
    """When model.snapshot_id is set, delete_model should also delete the snapshot row."""
    snapshot_id = uuid4()
    model = _make_model(snapshot_id=snapshot_id)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("services.model_service.get_async_db_session_ctx", return_value=mock_session),
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService.delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_awaited_once_with(model.snapshot_id)


@pytest.mark.anyio
async def test_delete_model_skips_snapshot_delete_when_no_snapshot_id() -> None:
    """When model.snapshot_id is None, snapshot repo delete should NOT be called."""
    model = _make_model(snapshot_id=None)

    mock_model_repo = AsyncMock()
    mock_snapshot_repo = AsyncMock()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("services.model_service.get_async_db_session_ctx", return_value=mock_session),
        patch("services.model_service.ModelRepository", return_value=mock_model_repo),
        patch("services.model_service.SnapshotRepository", return_value=mock_snapshot_repo),
        patch("services.model_service.shutil.rmtree"),
    ):
        await ModelService.delete_model(model)

    mock_model_repo.delete_by_id.assert_awaited_once_with(model.id)
    mock_snapshot_repo.delete_by_id.assert_not_awaited()
