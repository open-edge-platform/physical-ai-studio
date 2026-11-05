from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_robot_client_factory, get_robot_service, get_runtime_session_registry
from main import app
from robots.features import feature_names
from tests.workers.fake_robot import FakeRobot, sent_actions
from workers.robot_worker import RobotRecipe, claim_robot

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
JOINT_NAMES = ["shoulder_pan", "shoulder_lift"]


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
def driver(tmp_path: Path) -> FakeRobot:
    return FakeRobot(str(tmp_path / "actions.jsonl"), JOINT_NAMES, [12.3, 0.0])


@pytest.fixture
def factory(mock_robot_client_factory, driver: FakeRobot):
    mock_robot_client_factory.build_robot_driver = AsyncMock(return_value=(driver, object()))
    return mock_robot_client_factory


@pytest.fixture
def client(factory, registry) -> Iterator[TestClient]:
    app.dependency_overrides[get_runtime_session_registry] = lambda: registry
    app.dependency_overrides[get_robot_service] = lambda: _StubRobotService(_StubRobot())
    app.dependency_overrides[get_robot_client_factory] = lambda: factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_http_get_is_a_websocket_upgrade_stub(client: TestClient) -> None:
    response = client.get(_url())

    assert response.status_code == 426


def test_stream_emits_named_joint_features(client: TestClient, factory, driver: FakeRobot, tmp_path: Path) -> None:
    messages: list[dict] = []

    with client.websocket_connect(_url()) as websocket:
        while len(messages) < 2:
            messages.append(websocket.receive_json())

    events = {message["event"]: message for message in messages}
    expected_keys = feature_names(JOINT_NAMES)

    assert events["state"] == {"event": "state", "data": {"connected": True}}
    assert events["observation"]["event"] == "observation"
    assert list(events["observation"]["data"]) == expected_keys
    assert events["observation"]["data"] == {"shoulder_pan.pos": 12.3, "shoulder_lift.pos": 0.0}
    # Observe-only: nothing was ever sent to the robot.
    assert sent_actions(tmp_path / "actions.jsonl") == []


def test_stream_reports_connect_failure_as_an_error_frame(client: TestClient, driver: FakeRobot, factory) -> None:
    failing = FakeRobot(driver.log, JOINT_NAMES, [0.0, 0.0], fail_connect=True)
    factory.build_robot_driver = AsyncMock(return_value=(failing, object()))

    with client.websocket_connect(_url()) as websocket:
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_connection_failed"
    assert "Failed to open serial port" in payload["message"]


def test_stream_refuses_a_robot_that_is_already_held(client: TestClient, driver: FakeRobot) -> None:
    with claim_robot(RobotRecipe.from_driver(uuid4(), driver)), client.websocket_connect(_url()) as websocket:
        payload = websocket.receive_json()

    assert payload["event"] == "error"
    assert payload["error_code"] == "robot_device_already_owned"
    assert f"fake:{driver.log}" in payload["message"]


def test_stream_releases_the_robot_on_close(client: TestClient, driver: FakeRobot) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.receive_json()

    with claim_robot(RobotRecipe.from_driver(ROBOT_ID, driver)):
        pass
