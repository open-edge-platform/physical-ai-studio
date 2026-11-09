"""The dataset video endpoint serves files that grow while recording: LeRobot v3 appends new
episodes to the same mp4. Browsers must therefore check for a newer version instead of reusing
a cached copy, which would end where the file ended when it was first fetched."""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_dataset_service
from main import app
from schemas import Dataset
from settings import Settings

VIDEO_PATH = "videos/observation.images.gripper/chunk-000/file-000.mp4"


class _StubDatasetService:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    async def get_dataset_by_id(self, dataset_id):
        return self._dataset


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Dataset]:
    settings = Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))
    monkeypatch.setattr("schemas.dataset.get_settings", lambda: settings)
    dataset = Dataset(id=uuid4(), name="Hackathon", default_task="", project_id=uuid4(), environment_id=uuid4())
    video = Path(dataset.path) / VIDEO_PATH
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00" * 4096)
    app.dependency_overrides[get_dataset_service] = lambda: _StubDatasetService(dataset)
    yield dataset
    app.dependency_overrides.clear()


def _video_url(dataset: Dataset) -> str:
    return f"/api/dataset/{dataset.id}/video/{VIDEO_PATH}"


def test_video_must_be_revalidated_before_reuse(dataset: Dataset) -> None:
    response = TestClient(app).get(_video_url(dataset))

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert "no-cache" in response.headers["cache-control"]


def test_range_request_must_be_revalidated_before_reuse(dataset: Dataset) -> None:
    # Browsers fetch video with range requests and cache the partial responses too.
    response = TestClient(app).get(_video_url(dataset), headers={"Range": "bytes=0-"})

    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert "no-cache" in response.headers["cache-control"]


def test_etag_changes_when_a_recording_grows_the_file(dataset: Dataset) -> None:
    client = TestClient(app)
    before = client.get(_video_url(dataset), headers={"Range": "bytes=0-"})

    with (Path(dataset.path) / VIDEO_PATH).open("ab") as video:
        video.write(b"\x01" * 4096)
    after = client.get(_video_url(dataset), headers={"Range": "bytes=0-"})

    assert after.headers["etag"] != before.headers["etag"]
    assert after.headers["content-range"] == "bytes 0-8191/8192"


@pytest.mark.parametrize("video_path", ["%2e%2e/secret.mp4", "videos/%2e%2e/%2e%2e/secret.mp4", "/etc/passwd"])
def test_video_rejects_unsafe_paths(dataset: Dataset, video_path: str) -> None:
    response = TestClient(app).get(f"/api/dataset/{dataset.id}/video/{video_path}")

    assert response.status_code == 403


@pytest.mark.parametrize("outside_dir", ["outside", "{dataset_id}-other"])
def test_video_rejects_symlinks_outside_dataset(dataset: Dataset, outside_dir: str) -> None:
    dataset_path = Path(dataset.path)
    outside = dataset_path.parent / outside_dir.format(dataset_id=dataset.id) / "secret.mp4"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"private")
    (dataset_path / "link.mp4").symlink_to(outside)

    response = TestClient(app).get(f"/api/dataset/{dataset.id}/video/link.mp4")

    assert response.status_code == 403


def test_video_allows_symlinks_within_dataset(dataset: Dataset) -> None:
    dataset_path = Path(dataset.path)
    (dataset_path / "link.mp4").symlink_to(dataset_path / VIDEO_PATH)

    response = TestClient(app).get(f"/api/dataset/{dataset.id}/video/link.mp4")

    assert response.status_code == 200
    assert response.content == b"\x00" * 4096
    assert "no-cache" in response.headers["cache-control"]


@pytest.mark.parametrize("video_path", ["missing.mp4", "videos"])
def test_video_returns_not_found_for_missing_files_and_directories(dataset: Dataset, video_path: str) -> None:
    response = TestClient(app).get(f"/api/dataset/{dataset.id}/video/{video_path}")

    assert response.status_code == 404
