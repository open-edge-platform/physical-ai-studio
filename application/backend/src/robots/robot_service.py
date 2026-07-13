from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_environment_repo import ProjectEnvironmentRepository
from repositories.project_robot_repo import ProjectRobotRepository
from robots.discovery.manager import DiscoveryManager
from schemas.robot import Robot, RobotWithConnectionState, RobotWithConnectionStateAdapter


class RobotService:
    @staticmethod
    async def get_robot_list(project_id: UUID) -> list[Robot]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            return await repo.get_all()

    @staticmethod
    async def find_online_robots(project_id: UUID) -> list[RobotWithConnectionState]:
        robots = await RobotService.get_robot_list(project_id)
        discovery = DiscoveryManager()
        await discovery.refresh_hardware_ports()

        results: list[RobotWithConnectionState] = []

        for robot in robots:
            is_online = await discovery.is_robot_online(robot)

            results.append(
                RobotWithConnectionStateAdapter.validate_python(
                    {
                        **robot.model_dump(),
                        "connection_status": "online" if is_online else "offline",
                    }
                )
            )

        return results

    @staticmethod
    async def get_robot_by_id(project_id: UUID, robot_id: UUID) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            robot = await repo.get_by_id(robot_id)

            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, robot_id)

            return robot

    @staticmethod
    async def create_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            return await repo.save(robot)

    @staticmethod
    async def update_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            return await repo.update(robot, partial_update=robot.model_dump(exclude={"id"}))

    @staticmethod
    async def delete_robot(project_id: UUID, robot_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)

            robot = await repo.get_by_id(robot_id)
            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

            try:
                await repo.delete_by_id(robot_id)
            except IntegrityError as e:
                await session.rollback()  # just in case the session is in a bad state after the IntegrityError
                env_repo = ProjectEnvironmentRepository(session, project_id)
                environment_names = await env_repo.find_environment_names_using_robot(robot_id)
                if environment_names:
                    raise ResourceInUseError(
                        ResourceType.ROBOT,
                        str(robot_id),
                        message=(
                            f"Robot '{robot.name}' cannot be deleted because it is used in environment(s): "
                            f"{', '.join(environment_names)}. Remove it from those environments first."
                        ),
                    )
                raise e
