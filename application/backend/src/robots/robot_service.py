from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_robot_repo import ProjectRobotRepository
from robots.discovery.manager import DiscoveryManager
from schemas.robot import Robot, RobotType, RobotWithConnectionState


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
                RobotWithConnectionState(
                    **robot.model_dump(),
                    connection_status="online" if is_online else "offline",
                )
            )

        return results

    @staticmethod
    async def get_robot_by_id(project_id: UUID, robot_id: UUID) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            robot = await repo.get_by_id(robot_id)

            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, project_id)

            return robot

    @staticmethod
    async def create_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)

            if robot.type in { RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
                robot.connection_string = ""

            return await repo.save(robot)

    @staticmethod
    async def update_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)

            if robot.type in { RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
                robot.connection_string = ""

            return await repo.update(robot, partial_update=robot.model_dump(exclude={"id"}))

    @staticmethod
    async def delete_robot(project_id: UUID, robot_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)

            robot = await repo.get_by_id(robot_id)
            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

            await repo.delete_by_id(robot_id)
