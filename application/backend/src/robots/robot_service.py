from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_environment_repo import ProjectEnvironmentRepository
from repositories.project_robot_repo import ProjectRobotRepository
from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import Robot, RobotWithConnectionState, RobotWithConnectionStateAdapter
from utils.serial_robot_tools import RobotConnectionManager


class RobotService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _repo(self, project_id: UUID) -> ProjectRobotRepository:
        return ProjectRobotRepository(self.session, project_id)

    async def get_robot_list(self, project_id: UUID) -> list[Robot]:
        return await self._repo(project_id).get_all()

    async def find_online_robots(self, project_id: UUID) -> list[RobotWithConnectionState]:
        robots = await self.get_robot_list(project_id)

        # Single serial port scan shared across all probes
        manager = RobotConnectionManager()
        await manager.find_robots()

        registry = RobotCatalogRegistry()

        results: list[RobotWithConnectionState] = []

        for robot in robots:
            definition = registry.get_definition(robot.type)
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

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Robot:
        robot = await self._repo(project_id).get_by_id(robot_id)

        if robot is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot_id)

        return robot

    async def create_robot(self, project_id: UUID, robot: Robot) -> Robot:
        return await self._repo(project_id).save(robot)

    async def update_robot(self, project_id: UUID, robot: Robot) -> Robot:
        return await self._repo(project_id).update(robot, partial_update=robot.model_dump(exclude={"id"}))

    async def delete_robot(self, project_id: UUID, robot_id: UUID) -> None:
        repo = self._repo(project_id)

        robot = await repo.get_by_id(robot_id)
        if robot is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

        try:
            await repo.delete_by_id(robot_id)
        except IntegrityError as e:
            await self.session.rollback()
            env_repo = ProjectEnvironmentRepository(self.session, project_id)
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
