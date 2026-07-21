import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from repositories.project_environment_repo import ProjectEnvironmentRepository
from schemas.robot import RobotType


def _make_robot_db_model(*, robot_id: UUID | None = None, name: str = "Robot") -> MagicMock:
    model = MagicMock()
    model.id = str(robot_id or uuid4())
    model.name = name
    model.type = str(RobotType.SO101_FOLLOWER)
    model.payload = {
        "connection_string": "/dev/ttyUSB0",
        "serial_number": "SO101-TEST-001",
    }
    model.active_calibration_id = None
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


def _make_camera_db_model(*, camera_id: UUID | None = None, name: str = "Camera") -> MagicMock:
    model = MagicMock()
    model.id = str(camera_id or uuid4())
    model.driver = "usb_camera"
    model.name = name
    model.fingerprint = "/dev/video0:0"
    model.hardware_name = "Test USB Camera"
    model.payload = {
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    model.created_at = datetime(2026, 1, 1)
    model.updated_at = datetime(2026, 1, 1)
    return model


def test_build_robots_with_teleoperators_skips_missing_robot_and_logs_warning() -> None:
    environment_id = uuid4()
    missing_robot_id = uuid4()

    dangling_link = MagicMock()
    dangling_link.robot = None
    dangling_link.robot_id = str(missing_robot_id)
    dangling_link.tele_operator_type = "none"
    dangling_link.tele_operator_robot_id = None
    dangling_link.tele_operator_robot = None

    valid_link = MagicMock()
    valid_link.robot = _make_robot_db_model(name="Follower")
    valid_link.robot_id = valid_link.robot.id
    valid_link.tele_operator_type = "none"
    valid_link.tele_operator_robot_id = None
    valid_link.tele_operator_robot = None

    with patch("repositories.project_environment_repo.logger.warning") as warning_mock:
        robots = ProjectEnvironmentRepository._build_robots_with_teleoperators(
            environment_id,
            [dangling_link, valid_link],
        )

    assert len(robots) == 1
    assert robots[0].robot.name == "Follower"
    assert robots[0].tele_operator.type == "none"
    warning_mock.assert_called_once()


def test_build_robots_with_teleoperators_keeps_teleoperator_id_when_eager_robot_missing() -> None:
    environment_id = uuid4()
    teleop_robot_id = uuid4()

    link = MagicMock()
    link.robot = _make_robot_db_model(name="Follower")
    link.robot_id = link.robot.id
    link.tele_operator_type = "robot"
    link.tele_operator_robot_id = str(teleop_robot_id)
    link.tele_operator_robot = None

    with patch("repositories.project_environment_repo.logger.warning") as warning_mock:
        robots = ProjectEnvironmentRepository._build_robots_with_teleoperators(environment_id, [link])

    assert len(robots) == 1
    assert robots[0].robot.name == "Follower"
    assert robots[0].tele_operator.type == "robot"
    assert robots[0].tele_operator.robot_id == teleop_robot_id
    assert robots[0].tele_operator.robot is None
    warning_mock.assert_called_once()


def test_get_by_id_with_relations_skips_missing_camera_and_logs_warning() -> None:
    project_id = uuid4()
    environment_id = uuid4()

    missing_camera_id = uuid4()
    dangling_camera_link = MagicMock()
    dangling_camera_link.camera = None
    dangling_camera_link.camera_id = str(missing_camera_id)

    valid_camera_link = MagicMock()
    valid_camera_link.camera = _make_camera_db_model(name="Front Camera")
    valid_camera_link.camera_id = valid_camera_link.camera.id

    environment_row = MagicMock()
    environment_row.id = str(environment_id)
    environment_row.name = "Test Environment"
    environment_row.robot_links = []
    environment_row.camera_links = [dangling_camera_link, valid_camera_link]
    environment_row.created_at = datetime(2026, 1, 1)
    environment_row.updated_at = datetime(2026, 1, 1)

    execute_result = MagicMock()
    execute_result.scalars.return_value.first.return_value = environment_row

    db_session = MagicMock()
    db_session.execute = AsyncMock(return_value=execute_result)

    repo = ProjectEnvironmentRepository(db_session, project_id)

    with patch("repositories.project_environment_repo.logger.warning") as warning_mock:
        environment = asyncio.run(repo.get_by_id_with_relations(environment_id))

    assert environment is not None
    assert environment.id == environment_id
    assert environment.name == "Test Environment"
    assert len(environment.cameras) == 1
    assert environment.cameras[0].name == "Front Camera"
    warning_mock.assert_called_once()
