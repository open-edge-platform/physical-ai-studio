from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_robot_repo import ProjectRobotRepository
from schemas.robot import Robot


class RobotService:
    @staticmethod
    async def get_robot_list(project_id: UUID) -> list[Robot]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRobotRepository(session, project_id)
            return await repo.get_all()

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

            await repo.delete_by_id(robot_id)
