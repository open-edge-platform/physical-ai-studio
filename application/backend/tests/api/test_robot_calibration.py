"""API tests for the robot calibration + robot update flow.

Regression coverage for the bug where ``PUT /api/projects/{pid}/robots/{rid}``
with ``active_calibration_id`` set would fail with a ``TypeError`` because
the repository update-refresh logic could not find an ``id`` on the item.
"""

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_robot_calibration_service, get_robot_service
from main import app
from schemas.calibration import Calibration
from schemas.robot import Robot, SO101Robot, SO101RobotPayload

# ---------------------------------------------------------------------------
# Minimal service stubs
# ---------------------------------------------------------------------------


class _StubRobotService:
    """In-memory stub for RobotService covering the endpoints under test."""

    def __init__(self) -> None:
        self._robots: dict[UUID, Robot] = {}

    async def create_robot(self, project_id: UUID, robot: Robot) -> Robot:
        self._robots[robot.id] = robot
        return robot

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Robot:
        return self._robots[robot_id]

    async def update_robot(self, project_id: UUID, robot: Robot) -> Robot:
        self._robots[robot.id] = robot
        return robot


class _StubRobotCalibrationService:
    """In-memory stub for RobotCalibrationService covering save_calibration."""

    def __init__(self) -> None:
        self._calibrations: dict[UUID, Calibration] = {}

    async def save_calibration(self, robot_id: UUID, calibration_data: Calibration) -> Calibration:
        calibration_data.robot_id = robot_id
        self._calibrations[calibration_data.id] = calibration_data
        return calibration_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_set_active_calibration_id_returns_updated_robot() -> None:
    """PUT with active_calibration_id should return 200 with the id set."""
    project_id = uuid4()
    robot_id = uuid4()
    calibration_id = str(uuid4())

    robot_stub = _StubRobotService()
    robot_stub._robots[robot_id] = SO101Robot(
        id=robot_id,
        name="Test Robot",
        type="SO101_Follower",
        payload=SO101RobotPayload(connection_string="", serial_number="SN-TEST-001"),
        active_calibration_id=None,
    )

    app.dependency_overrides[get_robot_service] = lambda: robot_stub

    try:
        client = TestClient(app)

        update_resp = client.put(
            f"/api/projects/{project_id}/robots/{robot_id}",
            json={
                "id": str(robot_id),
                "name": "Test Robot",
                "type": "SO101_Follower",
                "payload": {
                    "connection_string": "",
                    "serial_number": "SN-TEST-001",
                },
                "active_calibration_id": calibration_id,
            },
        )
        assert update_resp.status_code == 200, update_resp.text

        body = update_resp.json()
        assert body["id"] == str(robot_id)
        assert body["active_calibration_id"] == calibration_id

    finally:
        app.dependency_overrides.clear()


def test_update_robot_without_calibration_returns_null_active_calibration() -> None:
    """Robot update without calibration should keep active_calibration_id as None."""
    project_id = uuid4()
    robot_id = uuid4()

    robot_stub = _StubRobotService()
    robot_stub._robots[robot_id] = SO101Robot(
        id=robot_id,
        name="Robot Without Calibration",
        type="SO101_Follower",
        payload=SO101RobotPayload(connection_string="", serial_number="SN-TEST-002"),
        active_calibration_id=None,
    )

    app.dependency_overrides[get_robot_service] = lambda: robot_stub

    try:
        client = TestClient(app)

        update_resp = client.put(
            f"/api/projects/{project_id}/robots/{robot_id}",
            json={
                "id": str(robot_id),
                "name": "Renamed Robot",
                "type": "SO101_Follower",
                "payload": {
                    "connection_string": "",
                    "serial_number": "SN-TEST-002",
                },
                "active_calibration_id": None,
            },
        )
        assert update_resp.status_code == 200, update_resp.text

        body = update_resp.json()
        assert body["name"] == "Renamed Robot"
        assert body["active_calibration_id"] is None

    finally:
        app.dependency_overrides.clear()


def test_save_calibration_endpoint_persists_robot_id() -> None:
    """POST /calibrations should store the robot_id from the URL, not the payload."""
    project_id = uuid4()
    robot_id = uuid4()
    calibration_stub = _StubRobotCalibrationService()

    app.dependency_overrides[get_robot_calibration_service] = lambda: calibration_stub

    try:
        client = TestClient(app)

        calibration_id = str(uuid4())
        different_robot_id = str(uuid4())
        cal_resp = client.post(
            f"/api/projects/{project_id}/robots/{robot_id}/calibrations",
            json={
                "id": calibration_id,
                "robot_id": different_robot_id,
                "file_path": "/tmp/test.json",
                "values": {
                    "shoulder_pan": {
                        "id": 1,
                        "joint_name": "shoulder_pan",
                        "drive_mode": 0,
                        "homing_offset": 0,
                        "range_min": 0,
                        "range_max": 4095,
                    }
                },
            },
        )
        assert cal_resp.status_code == 200, cal_resp.text
        body = cal_resp.json()
        assert UUID(body["robot_id"]) == robot_id

    finally:
        app.dependency_overrides.clear()
