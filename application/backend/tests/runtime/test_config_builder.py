from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from physicalai.config import to_yaml, validate_config
from physicalai.runtime import RobotRuntime

from runtime.config_builder import build_runtime_config, runtime_config_change_me
from schemas import SerialPortInfo
from schemas.project_camera import CameraAdapter
from schemas.robot import RobotAdapter


class FakePortResolver:
    def __init__(self, *, stable: bool = True) -> None:
        self.stable = stable

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        return port_info.connection_string

    def resolve_serial_device(self, device: str) -> str:
        return "/dev/serial/by-id/test-robot" if self.stable else device

    def resolve_camera_device(self, device: str) -> str:
        return "/dev/v4l/by-id/test-camera" if self.stable else device


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


async def test_builder_emits_valid_runtime_recipe_and_round_trips() -> None:
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=_robot("leader"),
        cameras=[_camera()],
        fps=30,
        port_resolver=FakePortResolver(),
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


async def test_builder_marks_unstable_device_paths() -> None:
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=_robot("leader"),
        cameras=[_camera()],
        fps=30,
        port_resolver=FakePortResolver(stable=False),
    )

    assert runtime_config_change_me(document) == ["/dev/ttyACM0", "/dev/video0", "/dev/ttyACM0"]
