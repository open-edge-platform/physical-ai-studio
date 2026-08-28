from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import (
    get_camera_claim_registry,
    get_camera_service,
    get_project_service,
    get_robot_client_factory,
    get_robot_service,
)
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas.project_camera import CameraAdapter
from services.camera_claims import CameraClaim, CameraClaimRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
CAMERA_ID = uuid4()
FOREIGN_CAMERA_ID = uuid4()


class _StubRobot:
    def __init__(self) -> None:
        self.id = ROBOT_ID
        self.name = "Khaos"


class _StubProject:
    def __init__(self) -> None:
        self.id = PROJECT_ID
        self.name = "Alpha"


class _StubRobotService:
    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> _StubRobot:
        assert project_id == PROJECT_ID
        assert robot_id == ROBOT_ID
        return _StubRobot()


class _StubProjectService:
    async def get_project_by_id(self, project_id: UUID) -> _StubProject:
        assert project_id == PROJECT_ID
        return _StubProject()


class _StubCameraService:
    def __init__(self, camera: object | None = None) -> None:
        self.lookups: list[tuple[UUID, UUID]] = []
        self._camera = camera

    async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> object:
        self.lookups.append((project_id, camera_id))
        if self._camera is None:
            raise ResourceNotFoundError(ResourceType.CAMERA, str(camera_id))
        return self._camera


def _camera() -> object:
    return CameraAdapter.validate_python(
        {
            "id": str(CAMERA_ID),
            "driver": "usb_camera",
            "name": "front",
            "fingerprint": "cam-front",
            "hardware_name": None,
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


@pytest.fixture
def claims() -> CameraClaimRegistry:
    return CameraClaimRegistry()


@pytest.fixture
def camera_service() -> _StubCameraService:
    return _StubCameraService()


@pytest.fixture
def client(
    mock_robot_client_factory,
    camera_service: _StubCameraService,
    claims: CameraClaimRegistry,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_robot_service] = lambda: _StubRobotService()
    app.dependency_overrides[get_camera_service] = lambda: camera_service
    app.dependency_overrides[get_robot_client_factory] = lambda: mock_robot_client_factory
    app.dependency_overrides[get_project_service] = lambda: _StubProjectService()
    app.dependency_overrides[get_camera_claim_registry] = lambda: claims
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


def test_a_failed_connect_releases_the_claim(mock_robot_client_factory, claims: CameraClaimRegistry) -> None:
    camera = _camera()
    camera_service = _StubCameraService(camera)
    app.dependency_overrides[get_robot_service] = lambda: _StubRobotService()
    app.dependency_overrides[get_camera_service] = lambda: camera_service
    app.dependency_overrides[get_robot_client_factory] = lambda: mock_robot_client_factory
    app.dependency_overrides[get_project_service] = lambda: _StubProjectService()
    app.dependency_overrides[get_camera_claim_registry] = lambda: claims
    try:
        test_client = TestClient(app)
        url = f"/api/projects/{PROJECT_ID}/runtime/ws"
        with (
            patch("api.runtime_ws.build_runtime_config", return_value={"init_args": {}}),
            patch("api.runtime_ws.RuntimeSessionOwner") as owner_cls,
            patch("api.runtime_ws.RuntimeSessionClient"),
            test_client.websocket_connect(url) as websocket,
        ):
            owner = owner_cls.return_value
            owner.connect.side_effect = RuntimeError("spawn failed")
            owner.is_alive.return_value = False
            websocket.send_json({"follower_id": str(ROBOT_ID), "camera_ids": [str(CAMERA_ID)]})
            payload = websocket.receive_json()
    finally:
        app.dependency_overrides.clear()

    assert payload["event"] == "error"
    assert claims.holder_of("cam-front") is None


def test_waiter_releases_when_the_owner_dies() -> None:
    from api.runtime_ws import _release_claims_when_dead

    claims = CameraClaimRegistry()
    generation = claims.claim(
        [
            CameraClaim(
                fingerprint="cam-front",
                settings=(640, 480, 30),
                holder="rt-a",
                project_id=PROJECT_ID,
                project_name="Alpha",
            )
        ]
    )
    owner = MagicMock()
    owner.is_alive.side_effect = [True, False]

    async def _run() -> None:
        with patch("api.runtime_ws._CLAIM_POLL_INTERVAL_S", 0):
            await _release_claims_when_dead(owner, claims, "rt-a", generation)

    asyncio.run(_run())

    assert claims.holder_of("cam-front") is None
