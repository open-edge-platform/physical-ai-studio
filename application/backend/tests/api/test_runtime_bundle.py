from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from physicalai.config import Config

from api.dependencies import (
    get_environment_service,
    get_model_download_service,
    get_model_service,
    get_robot_client_factory,
)
from main import app
from runtime.config_builder import policy_source_fragment
from schemas.environment import EnvironmentWithRelations
from tests.api.test_model_download import _make_model, _StubModelService


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


def test_runtime_bundle_zip_contains_yaml_readme_and_exports(tmp_path, mocker) -> None:
    export_dir = tmp_path / "exports" / "torch"
    export_dir.mkdir(parents=True)
    (export_dir / "model.pt").write_text("weights")
    model = _make_model(tmp_path)
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
                export_dir="./exports/torch",
                backend="torch",
                device="cpu",
                task="pick",
            ),
        },
    ).to_dict()

    environment_service = SimpleNamespace(get_environment_by_id=mocker.AsyncMock(return_value=environment))
    mocker.patch("api.models.build_runtime_config", return_value=document)

    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_model_download_service] = get_model_download_service
    app.dependency_overrides[get_environment_service] = lambda: environment_service
    app.dependency_overrides[get_robot_client_factory] = lambda: mocker.MagicMock()
    try:
        client = TestClient(app)
        response = client.get(
            f"/api/models/{model.id}/runtime-bundle",
            params={
                "environment_id": str(environment.id),
                "backend": "torch",
                "device": "cpu",
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
        assert "exports/torch/model.pt" in names
        yaml_text = archive.read("runtime.yaml").decode()
        assert "physicalai.runtime.PolicySource" in yaml_text
        assert "physicalai.runtime.AsyncExecution" in yaml_text
        assert "./exports/torch" in yaml_text
        assert "pick" in yaml_text
        readme = archive.read("README.md").decode()
        assert "physicalai run --config runtime.yaml" in readme


def test_runtime_bundle_404_when_backend_missing(tmp_path, mocker) -> None:
    model = _make_model(tmp_path)
    app.dependency_overrides[get_model_service] = lambda: _StubModelService(model)
    app.dependency_overrides[get_environment_service] = lambda: mocker.MagicMock()
    app.dependency_overrides[get_robot_client_factory] = lambda: mocker.MagicMock()
    try:
        client = TestClient(app)
        response = client.get(
            f"/api/models/{model.id}/runtime-bundle",
            params={"environment_id": str(uuid4()), "backend": "torch", "device": "cpu"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
