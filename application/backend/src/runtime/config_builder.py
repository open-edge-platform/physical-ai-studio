from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from physicalai.capture import ColorMode, SharedCamera
from physicalai.config import Config, to_config, validate_config
from physicalai_studio_plugin import CatalogRobotFactory, SerialPortInfo, shared_robot_name

from runtime.features import sanitize_camera_name
from utils.camera_factory import build_camera_config, is_migrated

if TYPE_CHECKING:
    from schemas.project_camera import Camera
    from schemas.robot import Robot


class PortResolver(CatalogRobotFactory, Protocol):
    def resolve_serial_device(self, device: str) -> str:
        """Return a stable serial path when one is available."""

    def resolve_camera_device(self, device: str) -> str:
        """Return a stable camera path when one is available."""


class _StableRobotPortResolver:
    def __init__(self, resolver: PortResolver) -> None:
        self._resolver = resolver

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        port = await self._resolver.find_port(port_info)
        if port is None:
            port = port_info.connection_string
        return None if port is None else self._resolver.resolve_serial_device(port)


async def _shared_robot_config(robot: Robot, resolver: PortResolver) -> dict[str, Any]:
    registry = getattr(resolver, "catalog_registry", None)
    if registry is None:
        from robots.catalog.registry import RobotCatalogRegistry

        registry = RobotCatalogRegistry()
    definition = registry.get_definition(robot.type)
    if definition is None:
        raise ValueError(f"Robot type is not part of the catalog: {robot.type}")
    if definition.robot_builder is None:
        raise ValueError(f"Robot type {robot.type} has no robot builder")
    driver = await definition.robot_builder(robot, _StableRobotPortResolver(resolver))
    return Config(
        "physicalai.robot.SharedRobot",
        {
            "name": shared_robot_name(robot.id),
            "robot": to_config(driver).to_dict(),
        },
    ).to_dict()


def _shared_camera_config(camera: Camera, resolver: PortResolver) -> dict[str, Any]:
    if not is_migrated(camera.driver):
        raise ValueError(f"Camera driver {camera.driver!r} is not supported by the runtime")
    fingerprint = camera.fingerprint
    if fingerprint.startswith("/dev/video") and ":" in fingerprint:
        fingerprint = fingerprint.split(":")[0]
    device = resolver.resolve_camera_device(fingerprint) if camera.driver == "usb_camera" else None
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
    port_resolver: PortResolver,
) -> dict[str, Any]:
    """Assemble one physicalai runtime recipe from Studio database rows."""
    follower_config = await _shared_robot_config(follower, port_resolver)
    leader_config = None if leader is None else await _shared_robot_config(leader, port_resolver)

    camera_configs: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        key = sanitize_camera_name(camera.name)
        if key in camera_configs:
            raise ValueError(f"Camera names collide after sanitizing: {camera.name!r}")
        camera_configs[key] = _shared_camera_config(camera, port_resolver)

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
