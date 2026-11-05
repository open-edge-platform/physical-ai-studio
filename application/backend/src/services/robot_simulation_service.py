# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""What a simulated robot's simulation serves (viewer, speed, cameras), and adding its cameras."""

import asyncio
import shlex
from typing import Any, cast
from uuid import UUID, uuid5

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationCommandUnavailableError,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotServingError,
    SimulationNotSupportedError,
)
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.project_camera import Camera, IPCamera, IPCameraPayload, SimulationCamera, SimulationCameraPayload
from schemas.robot import UnavailableRobot
from schemas.simulation import (
    ProjectRobotSimulation,
    RobotSimulation,
    SimulationCameraInfo,
    SimulationCommand,
    SimulationStatus,
    simulation_proxy_path,
)
from services.project_camera_service import ProjectCameraService
from services.simulation_manager import SimulationManager
from services.simulation_proxy import is_proxied_http_path, local_base_url

PROBE_TIMEOUT_S = 1.0
SIMULATION_CAMERA_IDS = UUID("5d1b2a4e-7c1f-4e0a-9a3c-2f6b8d0e1a7c")
"""Namespace of the ids of simulation cameras: one id per project, robot and camera."""


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
    ) -> list[SimulationCamera]:
        """Add some of a running simulation's cameras to an environment, as simulation cameras.

        A camera the project has already for this robot's simulation camera is reused, under its own
        name. A new one is named from *camera_names* (by simulation camera), or ``"<robot name>
        <camera>"`` when that has none or a blank one. Returns the cameras for *names*.

        Raises:
            ResourceNotFoundError: The robot or the environment does not exist.
            SimulationNotSupportedError: The robot type has no simulation.
            SimulationNotRunningError: The robot's simulation does not run.
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
        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        simulation = await self._probe(robot, self._owner_name(robot))
        if not simulation.running:
            raise SimulationNotRunningError(
                f"The simulation {simulation.owner_name!r} is not running.", robot_name=robot.name
            )
        served = {camera.name for camera in simulation.cameras}
        unknown = [name for name in names if name not in served]
        if unknown:
            raise SimulationInvalidRequestError(
                f"The simulation has no camera named {', '.join(unknown)}; it has {', '.join(served) or 'none'}."
            )

        camera_service = ProjectCameraService(self._session, self._registry)
        known = {
            camera.payload.camera: camera
            for camera in await camera_service.get_camera_list(project_id)
            if isinstance(camera, SimulationCamera) and camera.payload.robot_id == robot_id
        }
        cameras: list[SimulationCamera] = []
        created: list[SimulationCamera] = []
        for name in dict.fromkeys(names):
            camera = known.get(name)
            if camera is None:
                camera = _simulation_camera(
                    camera_names.get(name, "").strip() or f"{robot.name} {name}", project_id, robot_id, name
                )
                created.append(camera)
            cameras.append(camera)
        try:
            # One transaction: if any of it fails (e.g. the environment was deleted meanwhile), no new
            # camera is kept. A new camera's id follows from the robot and the camera, so two additions
            # at the same time make one camera. Linked, not written back from a snapshot of the
            # environment: a camera that another request added in the meantime stays.
            camera_repository = ProjectCameraRepository(self._session, str(project_id))
            for camera in created:
                await camera_repository.add_if_absent(camera)
            await environments.link_cameras(environment_id, [camera.id for camera in cameras])
            # Read again after these writes, which hold the database's write lock: a deletion of the
            # robot that went first is seen here, and one that comes after deletes these cameras too.
            await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        # As stored: a camera another addition created first keeps the name it gave it.
        return [
            cast("SimulationCamera", await camera_service.get_camera_by_id(project_id, camera.id)) for camera in cameras
        ]

    async def resolve_cameras(self, project_id: UUID, cameras: list[Camera]) -> list[Camera]:
        """Return *cameras* with each simulation camera as an IP camera on the stream it has now.

        A simulation starts on another address every time, so the address is asked of the running
        simulation, with the size and frame rate it renders at. The camera keeps its id, name and
        fingerprint, which claims go by. Other cameras are returned as they are.

        Raises:
            ResourceNotFoundError: The robot of a simulation camera does not exist.
            SimulationNotRunningError: The simulation of a simulation camera does not run.
            SimulationNotServingError: The simulation has no camera of that name.
        """
        simulations: dict[UUID, tuple[Any, RobotSimulation]] = {}
        resolved: list[Camera] = []
        for camera in cameras:
            if not isinstance(camera, SimulationCamera):
                resolved.append(camera)
                continue
            robot_id = camera.payload.robot_id
            if robot_id not in simulations:
                robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
                simulations[robot_id] = (robot, await self._probe(robot, self._owner_name(robot)))
            robot, simulation = simulations[robot_id]
            if not simulation.running:
                raise SimulationNotRunningError(
                    f"Start the simulation of {robot.name!r} to use its camera {camera.name!r}.",
                    robot_name=robot.name,
                )
            source = next((info for info in simulation.cameras if info.name == camera.payload.camera), None)
            if source is None:
                raise SimulationNotServingError(
                    f"The simulation of {robot.name!r} has no camera {camera.payload.camera!r} now, "
                    f"for camera {camera.name!r}."
                )
            resolved.append(
                IPCamera(
                    id=camera.id,
                    name=camera.name,
                    fingerprint=camera.fingerprint,
                    hardware_name=camera.hardware_name,
                    payload=IPCameraPayload(
                        url=source.stream_url, width=source.width, height=source.height, fps=source.fps
                    ),
                )
            )
        return resolved

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


def _simulation_camera(name: str, project_id: UUID, robot_id: UUID, camera: str) -> SimulationCamera:
    return SimulationCamera(
        id=uuid5(SIMULATION_CAMERA_IDS, f"{project_id}/{robot_id}/{camera}"),
        name=name,
        fingerprint={"robot_id": str(robot_id), "camera": camera},
        # Required by the database; the camera form also uses the name when there is no hardware.
        hardware_name=camera,
        payload=SimulationCameraPayload(robot_id=robot_id, camera=camera),
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


def _cameras(http_url: str, cameras: object) -> list[SimulationCameraInfo]:
    if not isinstance(cameras, list):
        return []
    result: list[SimulationCameraInfo] = []
    for camera in cameras:
        if not isinstance(camera, dict) or camera.get("failed"):
            continue
        try:
            result.append(
                SimulationCameraInfo(
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
