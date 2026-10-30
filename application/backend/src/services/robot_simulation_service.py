# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""What a simulated robot's simulation serves (viewer, speed, cameras), and adding its cameras."""

import asyncio
import shlex
from typing import Any
from uuid import UUID, uuid4

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationCommandUnavailableError,
    SimulationInvalidRequestError,
    SimulationMovedError,
    SimulationNotRunningError,
    SimulationNotServingError,
    SimulationNotSupportedError,
)
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.project_camera import IPCamera, IPCameraPayload
from schemas.robot import UnavailableRobot
from schemas.simulation import (
    ProjectRobotSimulation,
    RobotSimulation,
    SimulationCamera,
    SimulationCommand,
    SimulationStatus,
    simulation_proxy_path,
)
from services.project_camera_service import ProjectCameraService
from services.simulation_manager import SimulationManager
from services.simulation_proxy import is_proxied_http_path, local_base_url

PROBE_TIMEOUT_S = 1.0


class RobotSimulationService:
    def __init__(
        self,
        session: AsyncSession,
        registry: RobotCatalogRegistry,
        manager: SimulationManager,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._manager = manager
        self._transport = transport

    async def get(self, project_id: UUID, robot_id: UUID) -> RobotSimulation:
        """Return the state of a simulated robot's simulation, probing it over HTTP.

        Raises:
            ResourceNotFoundError: The robot does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
        """
        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        return _with_proxy_urls(project_id, robot_id, await self._probe(robot, self._owner_name(robot)))

    async def command(self, project_id: UUID, robot_id: UUID, scene: str, seed: int | None) -> SimulationCommand:
        """Return the command that starts the robot's simulation in a terminal, outside Studio.

        The simulation plugin builds it from the robot's payload: its name and address, so that Studio
        attaches to that simulation as it is.

        Raises:
            ResourceNotFoundError: The robot does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
            SimulationCommandUnavailableError: The robot type's plugin offers no such command.
            SimulationInvalidRequestError: The scene or seed is not valid for the robot type, or the
                robot's payload has no address to start it at.
        """
        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        # `standalone_argv` is missing on physicalai-studio-plugin releases before it.
        standalone_argv = getattr(self._launch(robot), "standalone_argv", None)
        try:
            argv = None if standalone_argv is None else standalone_argv(scene, robot.payload, seed)
        except ValueError as error:
            raise SimulationInvalidRequestError(str(error)) from error
        if argv is None:
            raise SimulationCommandUnavailableError(
                f"The simulation of robot {robot.name!r} ({robot.type}) cannot be started outside Studio."
            )
        return SimulationCommand(argv=argv, command=shlex.join(argv))

    async def proxy_addresses(self, project_id: UUID, robot_id: UUID) -> tuple[str | None, str | None]:
        """Return where the robot's simulation serves its HTTP API and its viewer, for passing them on.

        The addresses Studio knows for that simulation: its own while it runs it, else the robot's
        address and the viewer the simulation there reports. Only addresses on this machine; ``None``
        for one it does not serve, or not here.

        Raises:
            ResourceNotFoundError: The robot does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
            SimulationNotServingError: The robot's simulation does not run.
        """
        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        owner_name = self._owner_name(robot)
        state = self._manager.state()
        if state.robot_id == robot_id and state.project_id == project_id and state.status is SimulationStatus.RUNNING:
            http_url, viewer_url = state.http_url, state.viewer_url
        else:
            simulation = await self._probe(robot, owner_name)
            if not simulation.running:
                raise SimulationNotServingError(f"The simulation {owner_name!r} is not running.")
            http_url, viewer_url = simulation.http_url, simulation.viewer_url
        return local_base_url(http_url), local_base_url(viewer_url)

    async def list_running(self, project_id: UUID) -> list[ProjectRobotSimulation]:
        """Return the running simulations of the project's simulated robots, probed in parallel."""
        robots = [
            robot
            for robot in await RobotService(self._session, self._registry).get_robot_list(project_id)
            if not isinstance(robot, UnavailableRobot)
            and getattr(self._registry.get_definition(robot.type), "simulation", None) is not None
        ]
        results = await asyncio.gather(*(self._probe_or_none(robot) for robot in robots))
        return [
            ProjectRobotSimulation(
                **_with_proxy_urls(project_id, robot.id, result).model_dump(),
                robot_id=robot.id,
                robot_name=robot.name,
                robot_type=robot.type,
            )
            for robot, result in zip(robots, results, strict=True)
            if result is not None and result.running
        ]

    async def _probe_or_none(self, robot: Any) -> RobotSimulation | None:
        # One robot with a bad name or address must not hide the others.
        try:
            return await self._probe(robot, self._owner_name(robot))
        except Exception:
            logger.exception("Could not check the simulation of robot {!r}", robot.name)
            return None

    async def _probe(self, robot: Any, owner_name: str) -> RobotSimulation:
        robot_id = robot.id
        state = self._manager.state()
        if state.robot_id == robot_id and state.status in {SimulationStatus.STARTING, SimulationStatus.STOPPING}:
            # Studio is starting or stopping it: it may already answer, but it is not ready.
            return RobotSimulation(running=False, owner_name=owner_name)
        studio = state.robot_id == robot_id and state.status is SimulationStatus.RUNNING
        http_url = state.http_url if studio else getattr(robot.payload, "http_url", None)
        if not isinstance(http_url, str):
            return RobotSimulation(running=False, owner_name=owner_name)

        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, transport=self._transport) as client:
            root, health, cameras = await asyncio.gather(
                _get_json(client, f"{http_url}/"),
                _get_json(client, f"{http_url}/health"),
                _get_json(client, f"{http_url}/cameras"),
            )
        answers = isinstance(root, dict) and root.get("service") == owner_name
        if not answers and not studio:
            return RobotSimulation(running=False, owner_name=owner_name)

        root = root if isinstance(root, dict) else {}
        health = health if isinstance(health, dict) else {}
        viewer_url = state.viewer_url if studio else root.get("viewer_url")
        return RobotSimulation(
            running=True,
            started_by="studio" if studio else "external",
            owner_name=owner_name,
            http_url=http_url,
            viewer_url=viewer_url if isinstance(viewer_url, str) else None,
            scene=_str_or_none(health.get("scene")) or _str_or_none(root.get("scene")),
            cameras=_cameras(http_url, cameras),
        )

    async def add_cameras(
        self,
        project_id: UUID,
        environment_id: UUID,
        robot_id: UUID,
        names: list[str],
        camera_names: dict[str, str] | None = None,
    ) -> list[IPCamera]:
        """Add some of a running simulation's cameras to an environment, as IP cameras on their streams.

        A project camera already on a stream is reused, under its own name, unless another robot's
        simulation added it. A reused camera of no robot becomes this robot's. A new one is named from
        *camera_names* (by simulation camera), or ``"<robot name> <camera>"`` when that has none or a
        blank one. Returns the cameras for *names*.

        Raises:
            ResourceNotFoundError: The robot or the environment does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
            SimulationNotRunningError: The robot's simulation does not run.
            SimulationMovedError: The simulation restarted on another address after it was probed.
            SimulationInvalidRequestError: The simulation has no camera with one of the names, or
                *camera_names* names a camera that is not added.
        """
        camera_names = camera_names or {}
        not_added = [name for name in camera_names if name not in names]
        if not_added:
            raise SimulationInvalidRequestError(f"Names given for cameras that are not added: {', '.join(not_added)}.")
        environments = ProjectEnvironmentRepository(self._session, project_id, self._registry)
        if await environments.get_by_id(environment_id) is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))
        simulation = await self.get(project_id, robot_id)
        if not simulation.running:
            raise SimulationNotRunningError(f"The simulation {simulation.owner_name!r} is not running.")
        by_name = {camera.name: camera for camera in simulation.cameras}
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise SimulationInvalidRequestError(
                f"The simulation has no camera named {', '.join(unknown)}; it has {', '.join(by_name) or 'none'}."
            )

        # Studio does not move the robot's simulation (and its cameras) in the meantime, and the
        # simulation must still be where it was probed: else the cameras would be saved on a stopped
        # simulation's address (that a replacement's relocation already went past).
        async with self._manager.updating_robot(project_id, robot_id):
            self._session.expire_all()  # re-read below, not from this session's cache
            robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
            if self._current_address(robot) != simulation.http_url:
                raise SimulationMovedError(simulation.owner_name)
            camera_service = ProjectCameraService(self._session, self._registry)
            on_stream = await _reusable_cameras(camera_service, project_id, robot_id)
            cameras: list[IPCamera] = []
            created: list[IPCamera] = []
            adopted: list[IPCamera] = []
            for name in dict.fromkeys(names):
                source = by_name[name]
                camera = on_stream.get(source.stream_url)
                if camera is None:
                    camera = _ip_camera(camera_names.get(name, "").strip() or f"{robot.name} {name}", source, robot_id)
                    created.append(camera)
                elif camera.payload.simulation_robot_id is None:
                    payload = camera.payload.model_copy(update={"simulation_robot_id": robot_id})
                    camera = camera.model_copy(update={"payload": payload})
                    adopted.append(camera)
                cameras.append(camera)
            try:
                # One transaction: a failure (e.g. the environment deleted meanwhile) keeps neither the
                # new cameras nor the adoptions. Appended, not written back from a snapshot of the
                # environment: a camera that another request added in the meantime stays.
                await environments.add_cameras(
                    environment_id,
                    [camera.id for camera in cameras],
                    new_cameras=created,
                    changed_cameras=adopted,
                )
            except Exception:
                await self._session.rollback()
                raise
            return cameras

    def _current_address(self, robot: Any) -> str | None:
        """Return where the robot's simulation is now: Studio's address while Studio runs it."""
        state = self._manager.state()
        if state.robot_id == robot.id and state.status is SimulationStatus.RUNNING:
            return state.http_url
        http_url = getattr(robot.payload, "http_url", None)
        return http_url if isinstance(http_url, str) else None

    def _owner_name(self, robot: Any) -> str:
        return self._launch(robot).owner_name(robot.payload)

    def _launch(self, robot: Any) -> Any:
        definition = None if isinstance(robot, UnavailableRobot) else self._registry.get_definition(robot.type)
        launch = getattr(definition, "simulation", None)
        if launch is None:
            raise SimulationNotSupportedError(f"Robot {robot.name!r} ({robot.type}) has no simulation.")
        return launch


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.debug("Simulation probe {} failed: {}", url, error)
        return None


async def _reusable_cameras(cameras: ProjectCameraService, project_id: UUID, robot_id: UUID) -> dict[str, IPCamera]:
    """Return the cameras an addition of the robot's cameras reuses, by stream URL.

    A camera of this robot on the same stream is reused, not duplicated (e.g. a second click, or a
    stale page), and so is one of no robot, which then becomes this robot's. A camera of another
    robot is not: its simulation ran on this port before, and moves it when it restarts.
    """
    on_stream: dict[str, IPCamera] = {}
    for camera in await cameras.get_camera_list(project_id):
        if not isinstance(camera, IPCamera) or camera.payload.simulation_robot_id not in {robot_id, None}:
            continue
        known = on_stream.get(camera.payload.url)
        if known is None or known.payload.simulation_robot_id is None:
            on_stream[camera.payload.url] = camera
    return on_stream


def _ip_camera(name: str, source: SimulationCamera, robot_id: UUID) -> IPCamera:
    return IPCamera(
        id=uuid4(),
        name=name,
        fingerprint={"url": source.stream_url},
        # Required by the database; the camera form also uses the name when there is no hardware.
        hardware_name=name,
        payload=IPCameraPayload(
            url=source.stream_url,
            width=source.width,
            height=source.height,
            fps=source.fps,
            # So that starting the simulation again moves this camera, and no other robot's.
            simulation_robot_id=robot_id,
        ),
    )


def _with_proxy_urls(project_id: UUID, robot_id: UUID, simulation: RobotSimulation) -> RobotSimulation:
    """Add the addresses through Studio the browser uses for the simulation's viewer and frames."""
    if not simulation.running or simulation.http_url is None:
        return simulation
    path = simulation_proxy_path(project_id, robot_id)
    cameras = []
    for camera in simulation.cameras:
        api_path = camera.snapshot_url.removeprefix(simulation.http_url)
        proxied = f"{path}/http{api_path}" if is_proxied_http_path(api_path) else None
        cameras.append(camera.model_copy(update={"snapshot_proxy_url": proxied}))
    return simulation.model_copy(
        update={
            "viewer_proxy_url": f"{path}/viewer/" if simulation.viewer_url is not None else None,
            "cameras": cameras,
        }
    )


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _cameras(http_url: str, cameras: object) -> list[SimulationCamera]:
    if not isinstance(cameras, list):
        return []
    result: list[SimulationCamera] = []
    for camera in cameras:
        if not isinstance(camera, dict) or camera.get("failed"):
            continue
        try:
            result.append(
                SimulationCamera(
                    name=camera["name"],
                    width=camera["width"],
                    height=camera["height"],
                    fps=camera["fps"],
                    source=camera.get("source", ""),
                    stream_url=f"{http_url}{camera['stream_url']}",
                    snapshot_url=f"{http_url}{camera['snapshot_url']}",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.debug("Ignoring a malformed simulation camera {}: {}", camera, error)
    return result
