from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_dataset_service
from main import app
from schemas import Dataset


class _StubDatasetService:
    """Minimal stub for DatasetService covering the endpoints under test."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        # Capture call arguments for assertion
        self.update_dataset_name_calls: list[dict] = []
        self.delete_dataset_calls: list[dict] = []

    async def update_dataset_name(self, dataset_id: UUID, name: str) -> Dataset:
        self.update_dataset_name_calls.append({"dataset_id": dataset_id, "name": name})
        return self._dataset.model_copy(update={"name": name})

    async def delete_dataset(self, dataset_id: UUID, remove_files: bool) -> None:
        self.delete_dataset_calls.append({"dataset_id": dataset_id, "remove_files": remove_files})


def _make_dataset() -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Original Name",
        default_task="Pick and place",
        path="/datasets/test",
        project_id=uuid4(),
        environment_id=uuid4(),
    )


# ---------------------------------------------------------------------------
# PUT /api/dataset/{dataset_id}
# ---------------------------------------------------------------------------


def test_update_dataset_name_returns_200_with_updated_payload() -> None:
    dataset = _make_dataset()
    stub = _StubDatasetService(dataset)
    app.dependency_overrides[get_dataset_service] = lambda: stub

    try:
        client = TestClient(app)
        response = client.put(
            f"/api/dataset/{dataset.id}",
            json={"name": "New Name"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "New Name"
    assert body["id"] == str(dataset.id)

    # Check that we called the dataset service properly
    assert len(stub.update_dataset_name_calls) == 1
    call = stub.update_dataset_name_calls[0]
    assert call["dataset_id"] == dataset.id
    assert call["name"] == "New Name"


# ---------------------------------------------------------------------------
# DELETE /api/dataset/{dataset_id}
# ---------------------------------------------------------------------------


def test_delete_dataset_returns_204_by_default() -> None:
    dataset = _make_dataset()
    stub = _StubDatasetService(dataset)
    app.dependency_overrides[get_dataset_service] = lambda: stub

    try:
        client = TestClient(app)
        response = client.delete(f"/api/dataset/{dataset.id}")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204

    # Check that we don't remove files by default
    assert len(stub.delete_dataset_calls) == 1
    call = stub.delete_dataset_calls[0]
    assert call["dataset_id"] == dataset.id
    assert call["remove_files"] is False


def test_delete_dataset_returns_204_with_remove_files_query() -> None:
    dataset = _make_dataset()
    stub = _StubDatasetService(dataset)
    app.dependency_overrides[get_dataset_service] = lambda: stub

    try:
        client = TestClient(app)
        response = client.delete(f"/api/dataset/{dataset.id}?remove_files=true")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 204

    # Check how we called the datasert service
    assert len(stub.delete_dataset_calls) == 1
    call = stub.delete_dataset_calls[0]
    assert call["dataset_id"] == dataset.id
    assert call["remove_files"] is True
