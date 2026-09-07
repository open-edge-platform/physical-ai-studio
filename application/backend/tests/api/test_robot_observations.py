from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from physicalai.robot import RobotDeviceAlreadyOwned

from api.dependencies import get_robot_client_factory, get_robot_service
from main import app
from runtime.features import feature_names

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
JOINT_NAMES = ["shoulder_pan", "shoulder_lift"]


@dataclass
class FakeObservation:
    joint_positions: np.ndarray
    timestamp: float = 1.0
    sensor_data: dict | None = None
    images: dict | None = None


@dataclass
class FakeSharedRobot:
    joint_names: list[str]
    observation: FakeObservation
    connect_error: Exception | None = None
    connect_calls: int = 0
    disconnect_calls: int = 0
    _connected: bool = field(default=False, init=False)

    def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        self._connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self._connected = False

    def get_observation(self) -> FakeObservation:
        return self.observation

    def is_connected(self) -> bool:
        return self._connected


class _StubRobot:
    def __init__(self, robot_id: UUID = ROBOT_ID) -> None:
        self.id = robot_id
        self.name = "Khaos"


class _StubRobotService:
    def __init__(self, robot: _StubRobot) -> None:
        self.robot = robot

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> _StubRobot:
        assert project_id == PROJECT_ID
        assert robot_id == self.robot.id
        return self.robot


def _url(*, fps: int | None = 30) -> str:
    path = f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/observations/ws"
    if fps is None:
        return path
    return f"{path}?fps={fps}"


@pytest.fixture
def shared_robot() -> FakeSharedRobot:
    return FakeSharedRobot(
        joint_names=list(JOINT_NAMES),
        observation=FakeObservation(joint_positions=np.array([12.3, 0.0])),
    )


@pytest.fixture
def factory(mock_robot_client_factory, shared_robot: FakeSharedRobot):
    mock_robot_client_factory.build_shared_robot = AsyncMock(return_value=(shared_robot, object()))
    return mock_robot_client_factory


@pytest.fixture
def client(factory) -> Iterator[TestClient]:
    app.dependency_overrides[get_robot_service] = lambda: _StubRobotService(_StubRobot())
    app.dependency_overrides[get_robot_client_factory] = lambda: factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_http_get_is_a_websocket_upgrade_stub(client: TestClient) -> None:
    response = client.get(_url())

    assert response.status_code == 426


def test_stream_emits_named_joint_features(client: TestClient, factory, shared_robot: FakeSharedRobot) -> None:
    messages: list[dict] = []

    with client.websocket_connect(_url()) as websocket:
        while len(messages) < 2:
            messages.append(websocket.receive_json())

    events = {message["event"]: message for message in messages}
    expected_keys = feature_names(JOINT_NAMES, include_velocities=False)

    assert events["state"] == {"event": "state", "data": {"connected": True}}
    assert events["observation"]["event"] == "observation"
    assert list(events["observation"]["data"]) == expected_keys
    assert events["observation"]["data"] == {"shoulder_pan.pos": 12.3, "shoulder_lift.pos": 0.0}
    factory.build.assert_not_called()
    factory.build_shared_robot.assert_called_once()
    assert shared_robot.connect_calls == 1


def test_stream_reports_connect_failure_as_an_error_frame(client: TestClient, shared_robot: FakeSharedRobot) -> None:
    shared_robot.connect_error = RobotDeviceAlreadyOwned(
        "ignored",
        phase="device_lock_contention",
        device_ids=("serial:ttyACM0",),
    )

    with client.websocket_connect(_url()) as websocket:
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_device_already_owned"
    assert "serial:ttyACM0" in payload["message"]
    assert shared_robot.disconnect_calls == 1


def test_stream_disconnects_the_shared_robot_on_close(client: TestClient, shared_robot: FakeSharedRobot) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.receive_json()

    assert shared_robot.disconnect_calls == 1


def test_stream_does_not_create_a_runtime_session(client: TestClient) -> None:
    # Patch both the definition and the name bound in the owner: an in-process
    # import of RuntimeProcessHost would otherwise miss the source-module patch.
    with (
        patch("runtime.transport.lock.SessionNameLock") as lock_cls,
        patch("runtime.hosts.process_host.RuntimeProcessHost") as host_src,
        patch("runtime.owner.RuntimeProcessHost") as host_bound,
        patch("runtime.owner.RuntimeSessionOwner") as owner_cls,
        client.websocket_connect(_url()) as websocket,
    ):
        websocket.receive_json()

    lock_cls.assert_not_called()
    host_src.assert_not_called()
    host_bound.assert_not_called()
    owner_cls.assert_not_called()
