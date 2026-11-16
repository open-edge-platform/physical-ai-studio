"""PATCH /api/dataset/{dataset_id}/episodes/{episode_index} changes the task an episode was recorded with."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_dataset_service, get_runtime_session_registry
from main import app
from runtime_session.dataset import editing
from runtime_session.registry import RuntimeSessionRegistry
from schemas import Dataset
from settings import Settings
from tests.lerobot_datasets import record_dataset


class _StubDatasetService:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    async def get_dataset_by_id(self, dataset_id: UUID) -> Dataset:
        return self._dataset


class _RecordingRegistry(RuntimeSessionRegistry):
    """A registry where one session has the dataset loaded for recording."""

    def __init__(self, dataset_id: UUID) -> None:
        super().__init__()
        self._dataset_id = dataset_id

    def holder_of_dataset(self, dataset_id: UUID) -> str | None:
        return "the runtime session for 'Desk'" if dataset_id == self._dataset_id else None


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return record_dataset(tmp_path_factory.mktemp("recorded") / "dataset", ["pick", "plcae"])


@pytest.fixture
def dataset(recorded: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Dataset]:
    settings = Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))
    monkeypatch.setattr("schemas.dataset.get_settings", lambda: settings)
    dataset = Dataset(id=uuid4(), name="Cubes", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    shutil.copytree(recorded, dataset.path)
    app.dependency_overrides[get_dataset_service] = lambda: _StubDatasetService(dataset)
    app.dependency_overrides[get_runtime_session_registry] = RuntimeSessionRegistry
    yield dataset
    app.dependency_overrides.clear()


def _url(dataset: Dataset, episode_index: int) -> str:
    return f"/api/dataset/{dataset.id}/episodes/{episode_index}"


def test_patch_returns_the_episode_and_later_reads_see_the_new_task(dataset: Dataset) -> None:
    client = TestClient(app)
    # Fill Studio's read cache first, so the test shows the edit refreshes it.
    assert [episode["tasks"] for episode in client.get(f"/api/dataset/{dataset.id}/episodes").json()] == [
        ["pick"],
        ["plcae"],
    ]

    response = client.patch(_url(dataset, 1), json={"task": " place "})

    assert response.status_code == 200
    body = response.json()
    assert body["episode_index"] == 1
    assert body["tasks"] == ["place"]
    assert len(body["actions"]) == body["length"]
    assert client.get(_url(dataset, 1)).json()["tasks"] == ["place"]
    assert [episode["tasks"] for episode in client.get(f"/api/dataset/{dataset.id}/episodes").json()] == [
        ["pick"],
        ["place"],
    ]


def test_patch_missing_episode_returns_404(dataset: Dataset) -> None:
    response = TestClient(app).patch(_url(dataset, 5), json={"task": "place"})

    assert response.status_code == 404


@pytest.mark.parametrize("payload", [{"task": ""}, {"task": "   "}, {}])
def test_patch_requires_a_task(dataset: Dataset, payload: dict) -> None:
    response = TestClient(app).patch(_url(dataset, 1), json=payload)

    assert response.status_code == 400  # Studio answers validation errors with 400


def test_patch_is_refused_while_a_session_records_into_the_dataset(dataset: Dataset) -> None:
    app.dependency_overrides[get_runtime_session_registry] = lambda: _RecordingRegistry(dataset.id)

    response = TestClient(app).patch(_url(dataset, 1), json={"task": "place"})

    assert response.status_code == 409
    assert "the runtime session for 'Desk'" in response.json()["message"]
    assert TestClient(app).get(_url(dataset, 1)).json()["tasks"] == ["plcae"]


def test_patch_is_refused_while_the_dataset_is_being_edited(dataset: Dataset) -> None:
    with editing(dataset.id):
        response = TestClient(app).patch(_url(dataset, 1), json={"task": "place"})

    assert response.status_code == 409
    assert "is being edited" in response.json()["message"]
    assert TestClient(app).get(_url(dataset, 1)).json()["tasks"] == ["plcae"]


def test_registry_names_the_session_that_has_the_dataset_open(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_id = uuid4()
    open_datasets: dict[UUID, list[str]] = {uuid4(): ["other"]}
    monkeypatch.setattr("runtime_session.dataset._OPEN", open_datasets)
    registry = RuntimeSessionRegistry()
    for session_id in ("idle", "other", "recording"):
        registry.add(SimpleNamespace(id=session_id, label=f"session {session_id}"))  # type: ignore[arg-type]
    assert registry.holder_of_dataset(dataset_id) is None

    open_datasets[dataset_id] = ["recording"]
    assert registry.holder_of_dataset(dataset_id) == "session recording"

    registry.remove(registry._sessions["recording"])  # closing, still copying back
    assert registry.holder_of_dataset(dataset_id) == "a runtime session that is closing"
