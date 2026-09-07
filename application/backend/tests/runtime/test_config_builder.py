from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import pytest
from physicalai.config import to_yaml, validate_config
from physicalai.runtime import RobotRuntime

from robots.robot_client_factory import RobotClientFactory
from runtime.config_builder import (
    POLICY_REQUEST_THRESHOLD,
    RUNTIME_FPS,
    build_runtime_config,
    policy_source_fragment,
    runtime_camera_keys,
    runtime_config_change_me,
    runtime_export_readme,
    runtime_identity_digest,
)
from schemas import SerialPortInfo
from schemas.project_camera import CameraAdapter
from schemas.robot import RobotAdapter


class FakePortFinder:
    def __init__(self, *, discovers: bool = True) -> None:
        self._discovers = discovers

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        return port_info.connection_string if self._discovers else None


def _robot_factory(*, discovers: bool = True) -> RobotClientFactory:
    return RobotClientFactory(robot_manager=FakePortFinder(discovers=discovers))  # type: ignore[arg-type]


def _stub_device_paths(mocker: Any) -> None:
    mocker.patch("robots.robot_client_factory.resolve_serial_device", return_value="/dev/serial/by-id/test-robot")


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


def _camera(*, name: str = "Overhead Camera") -> Any:
    return CameraAdapter.validate_python(
        {
            "id": str(uuid4()),
            "driver": "usb_camera",
            "name": name,
            "fingerprint": {"serial": "test-camera"},
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
    assert camera_device == {"serial": "test-camera"}
    calibration = document["init_args"]["robot"]["init_args"]["robot"]["init_args"]["calibration"]
    assert isinstance(calibration, dict)

    with NamedTemporaryFile(mode="w", suffix=".yaml") as config_file:
        config_file.write(to_yaml(document))
        config_file.flush()
        runtime = RobotRuntime.from_config(config_file.name)
    assert isinstance(runtime, RobotRuntime)


async def test_builder_marks_unstable_device_paths(mocker: Any) -> None:
    mocker.patch("robots.robot_client_factory.resolve_serial_device", side_effect=lambda device: device)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=_robot("leader"),
        cameras=[_camera()],
        fps=30,
        robot_factory=_robot_factory(),
    )

    assert runtime_config_change_me(document) == ["/dev/ttyACM0", "/dev/ttyACM0"]


async def test_builder_refuses_a_robot_that_is_not_attached() -> None:
    """A live session must not be handed a port nobody has seen."""
    with pytest.raises(ValueError, match="Could not resolve a serial port"):
        await build_runtime_config(
            follower=_robot("follower"),
            leader=None,
            cameras=[],
            fps=30,
            robot_factory=_robot_factory(discovers=False),
        )


async def test_export_keeps_the_stored_port_when_the_robot_is_absent(mocker: Any) -> None:
    """An exported config describes a rig that does not have to be plugged in."""
    mocker.patch("robots.robot_client_factory.resolve_serial_device", side_effect=lambda device: device)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[],
        fps=30,
        robot_factory=_robot_factory(discovers=False),
        allow_stored_port=True,
    )

    port = document["init_args"]["robot"]["init_args"]["robot"]["init_args"]["port"]
    assert port == "/dev/ttyACM0"
    assert runtime_config_change_me(document) == ["/dev/ttyACM0"]


async def test_change_me_lists_unstable_ports_not_the_accelerator(mocker: Any) -> None:
    mocker.patch("robots.robot_client_factory.resolve_serial_device", side_effect=lambda device: device)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[_camera()],
        fps=30,
        robot_factory=_robot_factory(discovers=False),
        allow_stored_port=True,
        action_source=policy_source_fragment(
            export_dir="./exports/openvino",
            backend="openvino",
            device="cpu",
        ),
    )

    unresolved = runtime_config_change_me(document)
    assert "/dev/ttyACM0" in unresolved
    assert "cpu" not in unresolved
    for accelerator in ("CPU", "GPU", "gpu"):
        document["init_args"]["action_source"]["init_args"]["model"]["init_args"]["device"] = accelerator
        assert accelerator not in runtime_config_change_me(document)


def _identity_document(
    *,
    fps: float = 30.0,
    cameras: dict[str, object] | None = None,
    leader: bool = False,
) -> dict[str, Any]:
    init_args: dict[str, Any] = {
        "robot": {"class_path": "tests.runtime.fakes.FakeRobot", "init_args": {"name": "follower"}},
        "cameras": cameras or {},
        "fps": fps,
    }
    if leader:
        init_args["action_source"] = {
            "class_path": "physicalai.runtime.TeleopSource",
            "init_args": {
                "leader": {"class_path": "tests.runtime.fakes.FakeRobot", "init_args": {"name": "leader"}},
            },
        }
    return {"class_path": "physicalai.runtime.RobotRuntime", "init_args": init_args}


def test_identity_digest_ignores_cameras() -> None:
    without_cameras = _identity_document()
    with_cameras = _identity_document(cameras={"overhead camera": {"class_path": "unused"}})

    assert runtime_identity_digest(without_cameras) == runtime_identity_digest(with_cameras)


def test_identity_digest_changes_with_the_leader() -> None:
    follower_only = _identity_document()
    with_leader = _identity_document(leader=True)

    assert runtime_identity_digest(follower_only) != runtime_identity_digest(with_leader)


def test_identity_digest_changes_with_fps() -> None:
    at_default = _identity_document(fps=RUNTIME_FPS)
    at_half = _identity_document(fps=15.0)

    assert runtime_identity_digest(at_default) != runtime_identity_digest(at_half)


async def test_camera_keys_are_sorted_sanitized_names(mocker: Any) -> None:
    _stub_device_paths(mocker)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[_camera(name="Zebra Cam"), _camera(name="Alpha/Cam")],
        robot_factory=_robot_factory(),
    )

    assert runtime_camera_keys(document) == ["alpha_cam", "zebra cam"]


async def test_cameras_validate_on_connect_and_never_overwrite(mocker: Any) -> None:
    _stub_device_paths(mocker)
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[_camera()],
        robot_factory=_robot_factory(),
    )

    shared_camera = document["init_args"]["cameras"]["overhead camera"]["init_args"]
    assert shared_camera["validate_on_connect"] is True
    assert shared_camera["overwrite_settings"] is False


def test_policy_source_fragment_matches_the_session_recipe() -> None:
    export = policy_source_fragment(
        export_dir="./exports/torch",
        backend="torch",
        device="cpu",
        task="pick up the cube",
    )
    session = policy_source_fragment(export_dir="/models/abc/exports/torch", backend="torch", device="cpu")

    assert export["class_path"] == "physicalai.runtime.PolicySource"
    assert export["init_args"]["execution"] == session["init_args"]["execution"]
    assert export["init_args"]["action_queue"] == session["init_args"]["action_queue"]
    assert export["init_args"]["execution"]["class_path"] == "physicalai.runtime.AsyncExecution"
    assert export["init_args"]["execution"]["init_args"]["request_threshold"] == POLICY_REQUEST_THRESHOLD
    assert "duration_frames" not in export["init_args"]["action_queue"]["init_args"]["smoother"]["init_args"]
    assert "policy_name" not in export["init_args"]["model"]["init_args"]
    assert export["init_args"]["model"]["init_args"]["export_dir"] == "./exports/torch"
    assert export["init_args"]["task"] == "pick up the cube"
    assert "task" not in session["init_args"]


def test_empty_task_is_omitted_from_the_fragment() -> None:
    fragment = policy_source_fragment(export_dir="./exports/torch", backend="torch", device="cpu", task="")
    assert "task" not in fragment["init_args"]


async def test_inference_export_document_uses_the_policy_fragment(mocker: Any) -> None:
    _stub_device_paths(mocker)
    fragment = policy_source_fragment(export_dir="./exports/torch", backend="torch", device="cpu", task="pick")
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[_camera()],
        robot_factory=_robot_factory(),
        allow_stored_port=True,
        action_source=fragment,
    )

    validate_config(document)
    source = document["init_args"]["action_source"]
    assert source["init_args"]["execution"] == fragment["init_args"]["execution"]
    assert source["init_args"]["action_queue"] == fragment["init_args"]["action_queue"]
    assert source["init_args"]["model"]["init_args"]["export_dir"] == "./exports/torch"
    assert "leader" not in source["init_args"]


def test_runtime_export_readme_lists_unresolved_paths() -> None:
    document = {
        "init_args": {"robot": {"init_args": {"name": "rt-follower"}}},
    }
    text = runtime_export_readme(document, unresolved=["/dev/ttyACM0"])
    assert "physicalai run --config runtime.yaml" in text
    assert "/dev/ttyACM0" in text
    assert "rt-follower" in text


async def test_inference_export_document_validates_with_a_policy_source(mocker: Any) -> None:
    _stub_device_paths(mocker)
    fragment = policy_source_fragment(export_dir="./exports/torch", backend="torch", device="cpu")
    document = await build_runtime_config(
        follower=_robot("follower"),
        leader=None,
        cameras=[],
        robot_factory=_robot_factory(),
        allow_stored_port=True,
        action_source=fragment,
    )
    validate_config(document)
    assert document["init_args"]["action_source"]["class_path"] == "physicalai.runtime.PolicySource"
