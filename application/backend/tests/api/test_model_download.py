# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_model_download_service, get_model_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas import Model


class _StubModelService:
    def __init__(self, model: Model | None):
        self._model = model

    async def get_model_by_id(self, model_id):
        if self._model is None:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return self._model


def _make_model(path: Path) -> Model:
    return Model(
        id=uuid4(),
        name="My Robot ACT Model @ v2",
        path=str(path),
        policy="act",
        properties={},
        project_id=uuid4(),
        dataset_id=uuid4(),
        snapshot_id=uuid4(),
    )


def test_model_download_returns_zip_archive_without_snapshot(tmp_path: Path) -> None:
    """Download should exclude snapshot_* directories by default."""
    model_dir = tmp_path / "model"
    (model_dir / "exports" / "torch").mkdir(parents=True)
    (model_dir / "snapshot_2026-03-25_14-30-45" / "data").mkdir(parents=True)

    (model_dir / "model.ckpt").write_text("checkpoint-data")
    (model_dir / "exports" / "torch" / "model.pt").write_text("exported-model")
    (model_dir / "snapshot_2026-03-25_14-30-45" / "data" / "episode.parquet").write_text("episode-data")

    model = _make_model(model_dir)

    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_model_download_service] = get_model_download_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/models/{model.id}/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert 'filename="My-Robot-ACT-Model-v2.zip"' in response.headers["content-disposition"]

    archive = io.BytesIO(response.content)
    assert zipfile.is_zipfile(archive)

    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
        assert "model.ckpt" in names
        assert "exports/torch/model.pt" in names
        # Snapshot should be excluded by default
        assert not any("snapshot_" in n for n in names)


def test_model_download_includes_snapshot_when_requested(tmp_path: Path) -> None:
    """Download with include_snapshot=true should include snapshot_* directories."""
    model_dir = tmp_path / "model"
    (model_dir / "exports" / "torch").mkdir(parents=True)
    (model_dir / "snapshot_2026-03-25_14-30-45" / "data").mkdir(parents=True)

    (model_dir / "model.ckpt").write_text("checkpoint-data")
    (model_dir / "exports" / "torch" / "model.pt").write_text("exported-model")
    (model_dir / "snapshot_2026-03-25_14-30-45" / "data" / "episode.parquet").write_text("episode-data")

    model = _make_model(model_dir)

    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_model_download_service] = get_model_download_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/models/{model.id}/download?include_snapshot=true")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    archive = io.BytesIO(response.content)
    with zipfile.ZipFile(archive) as zipped:
        names = sorted(zipped.namelist())
        assert "model.ckpt" in names
        assert "exports/torch/model.pt" in names
        # Snapshot should be included
        assert "snapshot_2026-03-25_14-30-45/data/episode.parquet" in names
        assert zipped.read("snapshot_2026-03-25_14-30-45/data/episode.parquet") == b"episode-data"


def test_model_download_returns_404_when_model_path_missing(tmp_path: Path) -> None:
    model = _make_model(tmp_path / "missing")

    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_model_download_service] = get_model_download_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/models/{model.id}/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "endpoint_not_found_response" in response.json()
