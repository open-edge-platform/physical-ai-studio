import socket
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from loguru import logger
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
from repositories.project_robot_repo import ProjectRobotRepository
from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import (
    Robot,
    RobotWithConnectionState,
    RobotWithConnectionStateAdapter,
    UnavailableRobot,
    UnavailableRobotWithConnectionState,
)
from utils.serial_robot_tools import RobotConnectionManager


class RobotService:
    def __init__(self, session: AsyncSession, catalog_registry: RobotCatalogRegistry) -> None:
        self.session = session
        self.catalog_registry = catalog_registry

    def _repo(self, project_id: UUID) -> ProjectRobotRepository:
        return ProjectRobotRepository(self.session, project_id, self.catalog_registry)

    async def get_robot_list(self, project_id: UUID) -> list[Robot | UnavailableRobot]:
        return await self._repo(project_id).get_all()

    async def find_online_robots(
        self, project_id: UUID
    ) -> list[RobotWithConnectionState | UnavailableRobotWithConnectionState]:
        robots = await self.get_robot_list(project_id)

        # Single serial port scan shared across all probes
        manager = RobotConnectionManager()
        await manager.find_robots()

        results: list[RobotWithConnectionState | UnavailableRobotWithConnectionState] = []

        for robot in robots:
            if isinstance(robot, UnavailableRobot):
                results.append(UnavailableRobotWithConnectionState(**robot.model_dump()))
                continue

            definition = self.catalog_registry.get_definition(robot.type)
            is_online = False
            if definition is not None and definition.probe is not None:
                is_online = await definition.probe.is_online(robot.payload, manager)

            results.append(
                RobotWithConnectionStateAdapter.validate_python(
                    {
                        **robot.model_dump(),
                        "connection_status": "online" if is_online else "offline",
                    }
                )
            )

        return results

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Robot | UnavailableRobot:
        robot = await self._repo(project_id).get_by_id(robot_id)

        if robot is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot_id)

        return robot

    async def create_robot(self, project_id: UUID, robot: Robot) -> Robot:
        return await self._repo(project_id).save(self._with_own_simulation(robot))

    def _with_own_simulation(self, robot: Robot) -> Robot:
        """Give a new simulated robot left on its plugin's defaults a simulation of its own.

        A simulated robot attaches to the simulation running under its owner ``name`` and finds it
        at ``http_url``. On the plugin's defaults, every robot of a type shares both, so a new robot
        would attach to the simulation of another robot, another Studio or a terminal. Each of the
        two that still has its default gets one of its own: the name the robot's whole id (ids made
        close in time can share a prefix), the address a port free now (starting the simulation
        moves on to another one if it is taken by then). The other one stays as it is, and so does
        every value after creation: to attach to a simulation on its defaults, edit the robot.
        """
        if getattr(self.catalog_registry.get_definition(robot.type), "simulation", None) is None:
            return robot
        payload = robot.payload
        update: dict[str, str] = {}
        name = _default_of(payload, "name")
        if name is not None and payload.name == name:
            update["name"] = f"{name}-{robot.id.hex}"
        http_url = _default_of(payload, "http_url")
        if http_url is not None and payload.http_url == http_url:
            update["http_url"] = _with_free_port(http_url)
        if not update:
            return robot
        logger.info("Robot {!r} gets its own simulation: {}", robot.name, update)
        return robot.model_copy(update={"payload": payload.model_copy(update=update)})

    async def update_robot(self, project_id: UUID, robot: Robot, *, commit: bool = True) -> Robot:
        """Update *robot*; with ``commit=False`` the caller commits it, with more, and gets *robot* back."""
        repo = self._repo(project_id)
        if not commit:
            await repo.stage_update(robot)
            return robot
        return await repo.update(robot, partial_update=robot.model_dump(exclude={"id"}))

    async def delete_robot(self, project_id: UUID, robot_id: UUID) -> None:
        repo = self._repo(project_id)

        robot = await repo.get_by_id(robot_id)
        if robot is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

        try:
            # One transaction, the robot first: deleting it takes the database's write lock, so the
            # cameras listed next include any that an addition committed before, and an addition
            # after it finds no robot. If the robot cannot be deleted, its cameras stay too.
            await repo.stage_delete(robot_id)
            await ProjectCameraRepository(self.session, str(project_id)).delete_simulation_cameras(robot_id)
            await self.session.commit()
        except IntegrityError as e:
            await self.session.rollback()
            env_repo = ProjectEnvironmentRepository(self.session, project_id, self.catalog_registry)
            environment_names = await env_repo.find_environment_names_using_robot(robot_id)
            if environment_names:
                raise ResourceInUseError(
                    ResourceType.ROBOT,
                    str(robot_id),
                    message=(
                        f"Robot '{robot.name}' cannot be deleted because it is used in environment(s): "
                        f"{', '.join(environment_names)}. Remove it from those environments first."
                    ),
                ) from e
            raise


def _default_of(payload: BaseModel, field: str) -> str | None:
    """Return the default of a payload's string *field*, or ``None`` when it has no such field or default."""
    info = type(payload).model_fields.get(field)
    default = None if info is None else info.get_default(call_default_factory=True)
    return default if isinstance(default, str) else None


def _with_free_port(http_url: str) -> str:
    """Return *http_url* on a port of its host that is free now; *http_url* when it has no such host."""
    url = urlsplit(http_url)
    if url.hostname is None:
        return http_url
    try:
        with socket.socket(socket.AF_INET6 if ":" in url.hostname else socket.AF_INET) as probe:
            probe.bind((url.hostname, 0))
            port = probe.getsockname()[1]
    except OSError:
        return http_url
    host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
    return urlunsplit(url._replace(netloc=f"{host}:{port}"))
