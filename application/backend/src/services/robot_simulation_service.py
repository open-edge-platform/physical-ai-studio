# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""What a simulated robot's simulation serves (viewer, speed, cameras), and adding its cameras."""

import asyncio
import contextlib
from typing import Any
from uuid import UUID, uuid4

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotSupportedError,
)
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.environment import CameraEnvironmentConfiguration
from schemas.project_camera import IPCamera, IPCameraPayload
from schemas.robot import UnavailableRobot
from schemas.simulation import ProjectRobotSimulation, RobotSimulation, SimulationCamera, SimulationStatus
from services.environment_service import EnvironmentService
from services.project_camera_service import ProjectCameraService
from services.simulation_manager import SimulationManager

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
        return await self._probe(robot, self._owner_name(robot))

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
                **result.model_dump(), robot_id=robot.id, robot_name=robot.name, robot_type=robot.type
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

        A project camera already on a stream is reused, under its own name. A new one is named from
        *camera_names* (by simulation camera), or ``"<robot name> <camera>"`` when that has none or a
        blank one. Returns the cameras for *names*.

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
        environment = await ProjectEnvironmentRepository(self._session, project_id, self._registry).get_by_id(
            environment_id
        )
        if environment is None:
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

        robot = await RobotService(self._session, self._registry).get_robot_by_id(project_id, robot_id)
        camera_service = ProjectCameraService(self._session, self._registry)
        # A camera on the same stream is reused, not duplicated (e.g. a second click, or a stale page).
        on_stream = {
            camera.payload.url: camera
            for camera in await camera_service.get_camera_list(project_id)
            if isinstance(camera, IPCamera)
        }
        in_environment = {configuration.camera_id for configuration in environment.cameras}
        cameras: list[IPCamera] = []
        created: list[IPCamera] = []
        try:
            for name in dict.fromkeys(names):
                source = by_name[name]
                camera = on_stream.get(source.stream_url)
                if camera is None:
                    camera = _ip_camera(camera_names.get(name, "").strip() or f"{robot.name} {name}", source)
                    await camera_service.create_camera(project_id, camera)
                    created.append(camera)
                cameras.append(camera)

            added = [
                CameraEnvironmentConfiguration(camera_id=camera.id)
                for camera in cameras
                if camera.id not in in_environment
            ]
            if added:
                await EnvironmentService(self._session, self._registry).update_environment(
                    project_id,
                    environment.model_copy(update={"cameras": [*environment.cameras, *added]}),
                )
        except Exception:
            # Each camera is committed on its own: remove the ones this call made, in a clean
            # transaction (the failure may have left the session's one unusable).
            await self._session.rollback()
            for camera in created:
                with contextlib.suppress(Exception):
                    await camera_service.delete_camera(project_id, camera.id)
            raise
        return cameras

    def _owner_name(self, robot: Any) -> str:
        definition = None if isinstance(robot, UnavailableRobot) else self._registry.get_definition(robot.type)
        launch = getattr(definition, "simulation", None)
        if launch is None:
            raise SimulationNotSupportedError(f"Robot {robot.name!r} ({robot.type}) has no simulation.")
        return launch.owner_name(robot.payload)


async def _get_json(client: httpx.AsyncClient, url: str) -> Any:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        logger.debug("Simulation probe {} failed: {}", url, error)
        return None


def _ip_camera(name: str, source: SimulationCamera) -> IPCamera:
    return IPCamera(
        id=uuid4(),
        name=name,
        fingerprint={"url": source.stream_url},
        # Required by the database; the camera form also uses the name when there is no hardware.
        hardware_name=name,
        payload=IPCameraPayload(url=source.stream_url, width=source.width, height=source.height, fps=source.fps),
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
