"""Build a runnable ``physicalai.runtime`` config from Studio database rows.

Studio does not run on ``physicalai.runtime`` itself. This module only writes the
recipe a user downloads to run an environment or model outside Studio, so it is
the one place that names ``physicalai.runtime`` classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from physicalai.capture import ColorMode, SharedCamera
from physicalai.config import Config, validate_config
from physicalai_studio_plugin import shared_robot_name

from robots.features import sanitize_camera_name
from robots.robot_client_factory import RobotClientFactory
from schemas.runtime_session import ExecutionConfig
from utils.camera_factory import build_camera_config, is_migrated

if TYPE_CHECKING:
    from physicalai_studio_plugin import CatalogRobotFactory, SerialPortInfo

    from schemas.project_camera import Camera
    from schemas.robot import Robot

RUNTIME_FPS = 30.0


class _StoredPortFallback:
    """Export-only port finder: keeps the registered port when a robot is absent.

    An exported config describes how to reconnect on another machine, so an
    unplugged robot must not fail the download — ``runtime_config_change_me``
    flags the unresolved path instead.
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
            "robot": Config.from_instance(driver).to_dict(),
        },
    ).to_dict()


def _shared_camera_config(camera: Camera) -> dict[str, Any]:
    if not is_migrated(camera.driver):
        raise ValueError(f"Camera driver {camera.driver!r} is not supported by the runtime")
    shared_camera = SharedCamera(
        camera=build_camera_config(camera),
        color_mode=ColorMode.RGB,
        validate_on_connect=True,
        overwrite_settings=False,
    )
    return Config.from_instance(shared_camera).to_dict()


def policy_source_fragment(
    *,
    export_dir: str,
    backend: str,
    device: str,
    task: str | None = None,
    execution: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Return the PolicySource recipe the inference export instantiates.

    ``execution`` is what a session's ``ModelControl`` runs with, so the export runs the same.

    Omit ``policy_name`` so the manifest is read. Omit ``duration_frames`` so
    ``LerpSmoother`` keeps its upstream default of 5.
    """
    execution = execution or ExecutionConfig()
    init_args: dict[str, Any] = {
        "model": Config(
            "physicalai.inference.InferenceModel",
            {"export_dir": export_dir, "backend": backend, "device": device},
        ).to_dict(),
        "execution": Config(
            "physicalai.runtime.AsyncExecution" if execution.mode == "async" else "physicalai.runtime.SyncExecution",
            {"request_threshold": execution.request_threshold},
        ).to_dict(),
        "action_queue": Config(
            "physicalai.runtime.ChunkedActionQueue",
            {"smoother": Config("physicalai.runtime.LerpSmoother", {}).to_dict()},
        ).to_dict(),
    }
    if task:
        init_args["task"] = task
    return Config("physicalai.runtime.PolicySource", init_args).to_dict()


def runtime_export_readme(document: dict[str, Any], *, unresolved: list[str]) -> str:
    """README for a portable inference zip, including CHANGE_ME paths."""
    robot_name = document["init_args"]["robot"]["init_args"]["name"]
    lines = [
        "# Studio runtime export",
        "",
        "Run from this directory so `./exports/<backend>` resolves:",
        "",
        "```bash",
        "physicalai run --config runtime.yaml --run.duration_s=60",
        "```",
        "",
    ]
    if unresolved:
        lines.extend(
            [
                "## CHANGE_ME",
                "",
                "These device paths are machine-specific. Replace them with the",
                "ports and cameras on this host:",
                "",
                *[f"- `{path}`" for path in unresolved],
                "",
            ]
        )
    lines.extend(
        [
            "The `device` field on the model is from the machine that produced this",
            "export. Change it if this host has no matching accelerator.",
            "",
            f"The robot name is `{robot_name}`. Two runs that share it collide on",
            "one host — that is the intended lock. Stop the other run first.",
            "",
        ]
    )
    return "\n".join(lines)


async def build_runtime_config(
    *,
    follower: Robot,
    leader: Robot | None,
    cameras: list[Camera],
    robot_factory: RobotClientFactory,
    fps: float = RUNTIME_FPS,
    allow_stored_port: bool = False,
    action_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one physicalai runtime recipe from Studio database rows.

    Serial ports are resolved against the devices attached right now, and an
    unresolvable one raises. Set ``allow_stored_port`` to keep the port stored
    at registration time instead: an exported config has to describe a rig that
    is not plugged in, while a session about to drive that rig must not guess.

    Pass ``action_source`` to pin a PolicySource fragment for a headless
    export. Without it, a leader becomes a TeleopSource.
    """
    port_finder: CatalogRobotFactory = _StoredPortFallback(robot_factory) if allow_stored_port else robot_factory
    follower_config = await _shared_robot_config(follower, robot_factory, port_finder)
    leader_config = None
    if leader is not None:
        try:
            leader_config = await _shared_robot_config(leader, robot_factory, port_finder)
        except ValueError as e:
            logger.debug(f"Leader could not connect: {e}")

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
    if action_source is not None:
        init_args["action_source"] = action_source
    elif leader_config is not None:
        init_args["action_source"] = {
            "class_path": "physicalai.runtime.TeleopSource",
            "init_args": {"leader": leader_config},
        }

    document = Config("physicalai.runtime.RobotRuntime", init_args).to_dict()
    validate_config(document)
    return cast("dict[str, Any]", document)


def _is_unstable_host_path(key: str, item: object) -> bool:
    """True when ``port`` or ``device`` is a machine-specific filesystem path.

    Accelerator names (``cpu``, ``CPU``, ``GPU``) also live under ``device``.
    Those are not host paths and must not be listed as CHANGE_ME ports.
    """
    if key not in {"port", "device"} or not isinstance(item, str):
        return False
    if item.startswith(("/dev/serial/by-id/", "/dev/v4l/by-id/")):
        return False
    return "/" in item or "\\" in item or item.upper().startswith("COM")


def runtime_config_change_me(document: dict[str, Any]) -> list[str]:
    """List unstable device paths that need editing on another machine."""
    paths: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if _is_unstable_host_path(key, item):
                    paths.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return paths
