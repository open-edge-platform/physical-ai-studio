from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_robot_repo import ProjectRobotRepository
from schemas.robot import Robot


class RobotService:
    @staticmethod
    async def get_robot_list(project_id: UUID) -> list[Robot]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session)
            return await repo.get_all(project_id)

    @staticmethod
    async def get_robot_by_id(project_id: UUID, robot_id: UUID) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session)
            robot = await repo.get_by_id(project_id, robot_id)

            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(project_id))

            return robot

    @staticmethod
    async def create_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session)
            return await repo.save(project_id, robot)

    @staticmethod
    async def update_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session)
            return await repo.update(project_id, robot.id, robot)

    @staticmethod
    async def delete_robot(project_id: UUID, robot_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session)

            robot = await repo.get_by_id(project_id, robot_id)
            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

            await repo.delete_by_id(project_id, robot_id)
