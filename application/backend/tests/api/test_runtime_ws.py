from __future__ import annotations

from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_camera_service, get_robot_client_factory, get_robot_service
from exceptions import ResourceNotFoundError, ResourceType
from main import app

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
FOREIGN_CAMERA_ID = uuid4()


class _StubRobot:
    def __init__(self) -> None:
        self.id = ROBOT_ID
        self.name = "Khaos"


class _StubRobotService:
    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> _StubRobot:
        assert project_id == PROJECT_ID
        assert robot_id == ROBOT_ID
        return _StubRobot()


class _StubCameraService:
    def __init__(self) -> None:
        self.lookups: list[tuple[UUID, UUID]] = []

    async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> None:
        self.lookups.append((project_id, camera_id))
        raise ResourceNotFoundError(ResourceType.CAMERA, str(camera_id))


@pytest.fixture
def camera_service() -> _StubCameraService:
    return _StubCameraService()


@pytest.fixture
def client(mock_robot_client_factory, camera_service: _StubCameraService) -> TestClient:
    app.dependency_overrides[get_robot_service] = lambda: _StubRobotService()
    app.dependency_overrides[get_camera_service] = lambda: camera_service
    app.dependency_overrides[get_robot_client_factory] = lambda: mock_robot_client_factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_runtime_ws_handshake_resolves_cameras_against_the_project(
    client: TestClient, camera_service: _StubCameraService
) -> None:
    url = f"/api/projects/{PROJECT_ID}/runtime/ws"

    with (
        patch("api.runtime_ws.RuntimeSessionOwner") as owner_cls,
        patch("api.runtime_ws.build_runtime_config") as build,
        client.websocket_connect(url) as websocket,
    ):
        websocket.send_json(
            {
                "follower_id": str(ROBOT_ID),
                "camera_ids": [str(FOREIGN_CAMERA_ID)],
            }
        )
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["error_code"] == "Camera_not_found"
    assert camera_service.lookups == [(PROJECT_ID, FOREIGN_CAMERA_ID)]
    build.assert_not_called()
    owner_cls.assert_not_called()
