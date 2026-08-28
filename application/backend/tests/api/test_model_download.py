# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi.testclient import TestClient
from physicalai.config import Config

from api.dependencies import (
    get_environment_service,
    get_model_download_service,
    get_model_service,
    get_robot_client_factory,
)
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from runtime.config_builder import policy_source_fragment
from schemas import Model
from schemas.environment import EnvironmentWithRelations


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


def _override_model_service(model: Model) -> None:
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_model_download_service] = get_model_download_service


def _override_export_download(model: Model, *, environment_service: object | None = None) -> None:
    _override_model_service(model)
    app.dependency_overrides[get_environment_service] = lambda: environment_service or MagicMock()
    app.dependency_overrides[get_robot_client_factory] = lambda: MagicMock()


def _environment() -> EnvironmentWithRelations:
    return EnvironmentWithRelations.model_validate(
        {
            "id": str(uuid4()),
            "name": "Test",
            "robots": [
                {
                    "robot": {
                        "id": str(uuid4()),
                        "name": "Follower",
                        "type": "SO101_Follower",
                        "payload": {"connection_string": "/dev/ttyACM0", "serial_number": "follower"},
                    },
                    "tele_operator": {"type": "none"},
                }
            ],
            "cameras": [],
        }
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
    _override_model_service(model)

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
    _override_model_service(model)

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
    _override_model_service(model)

    try:
        client = TestClient(app)
        response = client.get(f"/api/models/{model.id}/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "endpoint_not_found_response" in response.json()


def test_openvino_export_download_is_weights_only_without_recipe(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    export_dir = model_dir / "exports" / "openvino"
    export_dir.mkdir(parents=True)
    (export_dir / "model.xml").write_text("<net/>")
    model = _make_model(model_dir)
    # Weights-only must not require a robot manager / client factory.
    _override_model_service(model)
    app.dependency_overrides[get_environment_service] = lambda: MagicMock()

    try:
        client = TestClient(app)
        response = client.get(f"/api/models/{model.id}/exports/openvino/download")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert names == {"model.xml"}
        assert "runtime.yaml" not in names


def test_openvino_export_download_with_recipe_includes_yaml_and_exports(tmp_path, mocker) -> None:
    model_dir = tmp_path / "model"
    export_dir = model_dir / "exports" / "openvino"
    export_dir.mkdir(parents=True)
    (export_dir / "model.xml").write_text("<net/>")
    model = _make_model(model_dir)
    environment = _environment()
    document = Config(
        "physicalai.runtime.RobotRuntime",
        {
            "robot": {
                "class_path": "physicalai.robot.SharedRobot",
                "init_args": {"name": "rt-follower", "robot": {"class_path": "tests.runtime.fakes.FakeRobot"}},
            },
            "fps": 30.0,
            "action_source": policy_source_fragment(
                export_dir="./exports/openvino",
                backend="openvino",
                device="CPU",
                task="pick",
            ),
        },
    ).to_dict()

    environment_service = SimpleNamespace(get_environment_by_id=mocker.AsyncMock(return_value=environment))
    mocker.patch("api.models.build_runtime_config", return_value=document)
    _override_export_download(model, environment_service=environment_service)
    try:
        client = TestClient(app)
        response = client.get(
            f"/api/models/{model.id}/exports/openvino/download",
            params={
                "environment_id": str(environment.id),
                "device": "CPU",
                "task": "pick",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "studio-runtime-" in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
        assert "runtime.yaml" in names
        assert "README.md" in names
        assert "exports/openvino/model.xml" in names
        assert "model.xml" not in names
        yaml_text = archive.read("runtime.yaml").decode()
        assert "physicalai.runtime.PolicySource" in yaml_text
        assert "physicalai.runtime.AsyncExecution" in yaml_text
        assert "./exports/openvino" in yaml_text
        assert "pick" in yaml_text
        readme = archive.read("README.md").decode()
        assert "physicalai run --config runtime.yaml" in readme


def test_openvino_export_download_recipe_400_when_device_missing(tmp_path) -> None:
    model_dir = tmp_path / "model"
    (model_dir / "exports" / "openvino").mkdir(parents=True)
    (model_dir / "exports" / "openvino" / "model.xml").write_text("<net/>")
    model = _make_model(model_dir)
    _override_export_download(model)
    try:
        client = TestClient(app)
        response = client.get(
            f"/api/models/{model.id}/exports/openvino/download",
            params={"environment_id": str(uuid4())},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
