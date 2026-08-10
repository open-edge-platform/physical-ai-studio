from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from physicalai.config import to_yaml, validate_config
from physicalai.runtime import RobotRuntime

from robots.robot_client_factory import RobotClientFactory
from runtime.config_builder import build_runtime_config, runtime_config_change_me
from schemas import SerialPortInfo
from schemas.project_camera import CameraAdapter
from schemas.robot import RobotAdapter


class FakePortFinder:
    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        return port_info.connection_string


def _robot_factory() -> RobotClientFactory:
    return RobotClientFactory(robot_manager=FakePortFinder())  # type: ignore[arg-type]


def _stub_device_paths(mocker: Any) -> None:
    mocker.patch("runtime.config_builder.resolve_serial_device", return_value="/dev/serial/by-id/test-robot")
    mocker.patch("runtime.config_builder.resolve_camera_device", return_value="/dev/v4l/by-id/test-camera")


def _calibration() -> dict[str, dict[str, int]]:
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return {
        name: {"id": index + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for index, name in enumerate(names)
    }


def _robot(role: str) -> Any:
    return RobotAdapter.validate_python(
        {
            "id": str(uuid4()),
            "name": role,
            "type": f"SO101_{role.title()}",
            "payload": {
                "connection_string": "/dev/ttyACM0",
                "serial_number": "ABC123",
                "calibration": _calibration(),
            },
        }
    )


def _camera() -> Any:
    return CameraAdapter.validate_python(
        {
            "id": str(uuid4()),
            "driver": "usb_camera",
            "name": "Overhead Camera",
            "fingerprint": "/dev/video0:0",
            "hardware_name": "Camera",
            "payload": {"width": 640, "height": 480, "fps": 30},
        }
    )


async def test_builder_emits_valid_runtime_recipe_and_round_trips(mocker: Any) -> None:
    _stub_device_paths(mocker)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=_robot("leader"),
        cameras=[_camera()],
        fps=30,
        robot_factory=_robot_factory(),
    )

    validate_config(document)
    assert document["class_path"] == "physicalai.runtime.RobotRuntime"
    assert document["init_args"]["cameras"].keys() == {"overhead camera"}
    camera_device = document["init_args"]["cameras"]["overhead camera"]["init_args"]["camera"]["init_args"]["device"]
    assert camera_device == "/dev/v4l/by-id/test-camera"
    calibration = document["init_args"]["robot"]["init_args"]["robot"]["init_args"]["calibration"]
    assert isinstance(calibration, dict)

    with NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
        config_file.write(to_yaml(document))
        config_file.flush()
        runtime = RobotRuntime.from_config(config_file.name)
    assert isinstance(runtime, RobotRuntime)


async def test_builder_marks_unstable_device_paths(mocker: Any) -> None:
    mocker.patch("runtime.config_builder.resolve_serial_device", side_effect=lambda device: device)
    mocker.patch("runtime.config_builder.resolve_camera_device", side_effect=lambda device: device)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=_robot("leader"),
        cameras=[_camera()],
        fps=30,
        robot_factory=_robot_factory(),
    )

    assert runtime_config_change_me(document) == ["/dev/ttyACM0", "/dev/video0", "/dev/ttyACM0"]
