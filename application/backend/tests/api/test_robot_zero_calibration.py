from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from physicalai_studio_plugin import RobotCatalogDefinition, RobotZeroCalibration
from pydantic import BaseModel

from api.dependencies import get_robot_catalog_service, get_robot_client_factory
from main import app
from robots.catalog.registry import RobotCatalogRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.testclient import WebSocketTestSession

PROJECT_ID = uuid4()
ROBOT_TYPE = "Test_Calibrated_Follower"
JOINT_NAMES = ["shoulder_pan", "gripper"]


@dataclass
class FakeObservation:
    joint_positions: np.ndarray
    timestamp: float = 1.0
    sensor_data: dict | None = None
    images: dict | None = None


@dataclass
class FakeDriver:
    positions: list[float] = field(default_factory=lambda: [40.0, 20.0])
    zero_positions: list[float] = field(default_factory=lambda: [0.5, -0.2])
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    calls: list[str] = field(default_factory=list)
    fail_next_release: bool = False

    def connect(self) -> None:
        self.calls.append("connect")

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def get_observation(self) -> FakeObservation:
        return FakeObservation(joint_positions=np.array(self.positions))


class _TestPayload(BaseModel):
    connection_string: str = ""


def _zero_calibration() -> RobotZeroCalibration:
    async def release(robot: Any) -> None:
        robot.calls.append("release")
        if robot.fail_next_release:
            robot.fail_next_release = False
            raise RuntimeError("Servo did not unlock")

    async def set_zero(robot: Any) -> None:
        robot.calls.append("set_zero")
        robot.positions = list(robot.zero_positions)

    return RobotZeroCalibration(
        instructions="Fold the arm and close the gripper.",
        release=release,
        set_zero=set_zero,
        zero_tolerance_deg=1.0,
    )


class _FakeCatalogService:
    def __init__(self) -> None:
        self.registry = RobotCatalogRegistry()
        self.registry.register_robot(
            RobotCatalogDefinition(
                type=ROBOT_TYPE,
                display_name="Test Calibrated Follower",
                role="follower",
                robot_payload=_TestPayload,
                zero_calibration=_zero_calibration(),
            )
        )


@pytest.fixture
def driver() -> FakeDriver:
    return FakeDriver()


@pytest.fixture
def factory(mock_robot_client_factory, driver: FakeDriver):
    mock_robot_client_factory.build_robot_driver = AsyncMock(return_value=(driver, object()))
    return mock_robot_client_factory


@pytest.fixture
def client(factory) -> Iterator[TestClient]:
    catalog_service = _FakeCatalogService()
    app.dependency_overrides[get_robot_catalog_service] = lambda: catalog_service
    app.dependency_overrides[get_robot_client_factory] = lambda: factory
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _url() -> str:
    return f"/api/projects/{PROJECT_ID}/robots/zero-calibration/ws"


def _robot(robot_type: str = ROBOT_TYPE, payload: Any = None) -> dict[str, Any]:
    return {"id": str(uuid4()), "name": "Khaos", "type": robot_type, "payload": payload or {}}


def _receive_until(websocket: WebSocketTestSession, event: str, **match: Any) -> dict[str, Any]:
    for _ in range(100):
        message = websocket.receive_json()
        if message["event"] == event and all(message.get(key) == value for key, value in match.items()):
            return message
    raise AssertionError(f"No {event} event received")


def test_http_get_is_a_websocket_upgrade_stub(client: TestClient) -> None:
    assert client.get(_url()).status_code == 426


def test_start_releases_the_arm_and_streams_observations(client: TestClient, driver: FakeDriver) -> None:
    with client.websocket_connect(_url()) as websocket:
        _receive_until(websocket, "status", phase="waiting")
        websocket.send_json({"command": "start", "robot": _robot()})

        status = _receive_until(websocket, "status", phase="positioning")
        observation = _receive_until(websocket, "observation")

    assert status["message"] == "Fold the arm and close the gripper."
    assert observation["data"] == {"shoulder_pan.pos": 40.0, "gripper.pos": 20.0}
    assert driver.calls[:2] == ["connect", "release"]
    assert driver.calls[-1] == "disconnect"


def test_set_zero_reports_success_when_every_joint_reads_zero(client: TestClient, driver: FakeDriver) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "start", "robot": _robot()})
        _receive_until(websocket, "status", phase="positioning")
        websocket.send_json({"command": "set_zero"})

        result = _receive_until(websocket, "calibration_result")

    assert result == {
        "event": "calibration_result",
        "success": True,
        "joints": {"shoulder_pan.pos": 0.5, "gripper.pos": -0.2},
        "tolerance_deg": 1.0,
    }
    assert "set_zero" in driver.calls


def test_set_zero_reports_failure_when_a_joint_is_outside_tolerance(client: TestClient, driver: FakeDriver) -> None:
    driver.zero_positions = [0.5, 3.0]

    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "start", "robot": _robot()})
        _receive_until(websocket, "status", phase="positioning")
        websocket.send_json({"command": "set_zero"})

        result = _receive_until(websocket, "calibration_result")

    assert result["success"] is False
    assert result["joints"]["gripper.pos"] == 3.0


def test_failed_release_disconnects_so_start_can_be_retried(client: TestClient, driver: FakeDriver) -> None:
    driver.fail_next_release = True

    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "start", "robot": _robot()})
        error = _receive_until(websocket, "error")
        websocket.send_json({"command": "start", "robot": _robot()})
        _receive_until(websocket, "status", phase="positioning")

    assert error["message"] == "Servo did not unlock"
    assert driver.calls[:5] == ["connect", "release", "disconnect", "connect", "release"]


def test_set_zero_before_start_is_a_command_error(client: TestClient) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "set_zero"})

        error = _receive_until(websocket, "error")

    assert error["error_code"] == "command_error"


def test_start_rejects_robot_types_without_calibration(client: TestClient, factory) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "start", "robot": _robot("SO101_Follower", {"serial_number": "abc"})})

        error = _receive_until(websocket, "error")

    assert error["error_code"] == "zero_calibration_unsupported"
    factory.build_robot_driver.assert_not_called()


def test_start_rejects_invalid_robot_payload(client: TestClient, factory) -> None:
    with client.websocket_connect(_url()) as websocket:
        websocket.send_json({"command": "start", "robot": _robot(payload={"connection_string": 7})})

        error = _receive_until(websocket, "error")

    assert error["error_code"] == "invalid_config"
    factory.build_robot_driver.assert_not_called()
