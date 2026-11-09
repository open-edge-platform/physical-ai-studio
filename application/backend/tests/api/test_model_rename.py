from collections.abc import Iterator
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_model_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas import Model
from services import ModelService


@pytest.fixture
def model_service() -> Iterator[AsyncMock]:
    service = AsyncMock(spec=ModelService)
    app.dependency_overrides[get_model_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_model_service)


def test_rename_model(model_service: AsyncMock) -> None:
    model = Model(
        id=uuid4(),
        name="ACT",
        path="/models/original",
        policy="act",
        properties={},
        project_id=uuid4(),
    )
    model_service.get_model_by_id.return_value = model
    model_service.update_model.return_value = model.model_copy(update={"name": "My model"})

    response = TestClient(app).patch(f"/api/models/{model.id}", json={"name": "  My model  "})

    assert response.status_code == 200
    assert response.json()["name"] == "My model"
    assert response.json()["path"] == model.path
    model_service.get_model_by_id.assert_awaited_once_with(model.id)
    model_service.update_model.assert_awaited_once_with(model, {"name": "My model"})


@pytest.mark.parametrize("payload", [{}, {"name": ""}, {"name": "   "}, {"name": None}])
def test_rename_model_rejects_invalid_name(model_service: AsyncMock, payload: dict) -> None:
    response = TestClient(app).patch(f"/api/models/{uuid4()}", json=payload)

    assert response.status_code == 400
    model_service.update_model.assert_not_awaited()


def test_rename_missing_model(model_service: AsyncMock) -> None:
    model_id = uuid4()
    model_service.get_model_by_id.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

    response = TestClient(app).patch(f"/api/models/{model_id}", json={"name": "My model"})

    assert response.status_code == 404
    model_service.update_model.assert_not_awaited()
