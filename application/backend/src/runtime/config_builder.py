from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from physicalai.capture import ColorMode, SharedCamera
from physicalai.config import Config, to_config, validate_config
from physicalai_studio_plugin import shared_robot_name

from robots.robot_client_factory import RobotClientFactory
from runtime.features import sanitize_camera_name
from utils.camera_factory import build_camera_config, is_migrated
from utils.device_paths import resolve_camera_device

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobotFactory, SerialPortInfo

    from schemas.project_camera import Camera
    from schemas.robot import Robot


class _StoredPortFallback:
    """Export-only port finder: keeps the registered port when a robot is absent.

    An exported config describes how to reconnect on another machine, so an
    unplugged robot must not fail the download — ``runtime_config_change_me``
    flags the unresolved path instead. A live session must never do this: by
    then the stored path may belong to a different robot.
    """

    def __init__(self, port_finder: CatalogRobotFactory) -> None:
        self._port_finder = port_finder

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        port = await self._port_finder.find_port(port_info)
        return port if port is not None else port_info.connection_string


async def _shared_robot_config(
    robot: Robot, robot_factory: RobotClientFactory, port_finder: CatalogRobotFactory
) -> dict[str, Any]:
    driver, _ = await robot_factory.build_robot_driver(robot, port_finder)
    return Config(
        "physicalai.robot.SharedRobot",
        {
            "name": shared_robot_name(robot.id),
            "robot": to_config(driver).to_dict(),
        },
    ).to_dict()


def _shared_camera_config(camera: Camera) -> dict[str, Any]:
    if not is_migrated(camera.driver):
        raise ValueError(f"Camera driver {camera.driver!r} is not supported by the runtime")
    fingerprint = camera.fingerprint
    if fingerprint.startswith("/dev/video") and ":" in fingerprint:
        fingerprint = fingerprint.split(":")[0]
    device = resolve_camera_device(fingerprint) if camera.driver == "usb_camera" else None
    shared_camera = SharedCamera(
        camera=build_camera_config(camera, device=device),
        color_mode=ColorMode.RGB,
    )
    return to_config(shared_camera).to_dict()


async def build_runtime_config(
    *,
    follower: Robot,
    leader: Robot | None,
    cameras: list[Camera],
    fps: float,
    robot_factory: RobotClientFactory,
    allow_stored_port: bool = False,
) -> dict[str, Any]:
    """Assemble one physicalai runtime recipe from Studio database rows.

    Serial ports are resolved against the devices attached right now, and an
    unresolvable one raises. Set ``allow_stored_port`` to keep the port stored
    at registration time instead: an exported config has to describe a rig that
    is not plugged in, while a session about to drive that rig must not guess.
    """
    port_finder: CatalogRobotFactory = _StoredPortFallback(robot_factory) if allow_stored_port else robot_factory
    follower_config = await _shared_robot_config(follower, robot_factory, port_finder)
    leader_config = None if leader is None else await _shared_robot_config(leader, robot_factory, port_finder)

    camera_configs: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        key = sanitize_camera_name(camera.name)
        if key in camera_configs:
            raise ValueError(f"Camera names collide after sanitizing: {camera.name!r}")
        camera_configs[key] = _shared_camera_config(camera)

    init_args: dict[str, Any] = {
        "robot": follower_config,
        "cameras": camera_configs,
        "fps": float(fps),
    }
    if leader_config is not None:
        init_args["action_source"] = {
            "class_path": "physicalai.runtime.TeleopSource",
            "init_args": {"leader": leader_config},
        }

    document = Config("physicalai.runtime.RobotRuntime", init_args).to_dict()
    validate_config(document)
    return cast("dict[str, Any]", document)


def runtime_config_digest(document: dict[str, Any]) -> str:
    """Identify one rig configuration, so a client cannot attach to a different one."""
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def runtime_config_change_me(document: dict[str, Any]) -> list[str]:
    """List unstable device paths that need editing on another machine."""
    paths: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key in {"port", "device"}
                    and isinstance(item, str)
                    and not item.startswith(("/dev/serial/by-id/", "/dev/v4l/by-id/"))
                ):
                    paths.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return paths
