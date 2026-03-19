import io
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_dataset_download_service, get_dataset_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas import Dataset


class _StubDatasetService:
    def __init__(self, dataset: Dataset | None):
        self._dataset = dataset

    async def get_dataset_by_id(self, dataset_id):
        if self._dataset is None:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))
        return self._dataset


def _make_dataset(path: Path) -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Dataset Export @ 2026",
        default_task="Do the thing",
        path=str(path),
        project_id=uuid4(),
        environment_id=uuid4(),
    )


def test_dataset_download_returns_zip_archive(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "nested").mkdir(parents=True)
    (dataset_dir / "root.txt").write_text("root")
    (dataset_dir / "nested" / "child.txt").write_text("child")

    dataset = _make_dataset(dataset_dir)

    app.dependency_overrides[get_dataset_service] = lambda: _StubDatasetService(dataset)
    app.dependency_overrides[get_dataset_download_service] = get_dataset_download_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/dataset/{dataset.id}/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert 'filename="Dataset-Export-2026.zip"' in response.headers["content-disposition"]

    archive = io.BytesIO(response.content)
    assert zipfile.is_zipfile(archive)

    with zipfile.ZipFile(archive) as zipped:
        assert sorted(zipped.namelist()) == ["nested/child.txt", "root.txt"]
        assert zipped.read("root.txt") == b"root"
        assert zipped.read("nested/child.txt") == b"child"


def test_dataset_download_returns_404_when_dataset_path_missing(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path / "missing")

    app.dependency_overrides[get_dataset_service] = lambda: _StubDatasetService(dataset)
    app.dependency_overrides[get_dataset_download_service] = get_dataset_download_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/dataset/{dataset.id}/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "endpoint_not_found_response" in response.json()
