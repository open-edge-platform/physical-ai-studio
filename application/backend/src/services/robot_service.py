import json
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    delete,
    select,
    update,
)
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from schemas.robot import Robot, RobotCamera

Base = declarative_base()


class ProjectRobot(Base):
    __tablename__ = "project_robots"

    id = Column(Text, primary_key=True)
    project_id = Column(
        Text,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(255), nullable=False)
    serial_id = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    cameras = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class RobotRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_pydantic(self, db_robot: ProjectRobot) -> Robot:
        """Convert database model to Pydantic model."""

        # Parse cameras JSON string back to list
        cameras_data = []
        try:
            if isinstance(db_robot.cameras, str):
                cameras_data = json.loads(db_robot.cameras)
            else:
                cameras_data = db_robot.cameras
        except (json.JSONDecodeError, TypeError):
            cameras_data = []

        cameras = [RobotCamera(**camera) for camera in cameras_data]

        return Robot(
            id=db_robot.id,
            name=db_robot.name,
            serial_id=db_robot.serial_id,
            type=db_robot.type,
            cameras=cameras,
            created_at=db_robot.created_at,
            updated_at=db_robot.updated_at,
        )

    def _to_db_dict(self, robot: Robot, project_id: UUID) -> dict[str, Any]:
        """Convert Pydantic model to database dictionary."""
        cameras_json = json.dumps([camera.model_dump() for camera in robot.cameras])

        return {
            "id": str(robot.id),
            "project_id": str(project_id),
            "name": robot.name,
            "serial_id": robot.serial_id,
            "type": robot.type.value,
            "cameras": cameras_json,
        }

    async def get_all(self, project_id: UUID) -> list[Robot]:
        """Get all robots for a project."""
        stmt = select(ProjectRobot).where(ProjectRobot.project_id == str(project_id))
        result = await self.db.execute(stmt)
        db_robots = result.scalars().all()
        return [self._to_pydantic(db_robot) for db_robot in db_robots]

    async def get_by_id(self, project_id: UUID, robot_id: UUID) -> Robot | None:
        """Get a robot by ID within a project."""
        stmt = select(ProjectRobot).where(
            ProjectRobot.project_id == str(project_id), ProjectRobot.id == str(robot_id)
        )
        result = await self.db.execute(stmt)
        db_robot = result.scalar_one_or_none()

        if db_robot is None:
            return None

        return self._to_pydantic(db_robot)

    async def save(self, project_id: UUID, robot: Robot) -> Robot:
        """Save a new robot."""
        robot_data = self._to_db_dict(robot, project_id)

        db_robot = ProjectRobot(**robot_data)
        self.db.add(db_robot)
        await self.db.commit()
        await self.db.refresh(db_robot)

        return self._to_pydantic(db_robot)

    async def update(self, project_id: UUID, robot_id: UUID, robot: Robot) -> Robot:
        """Update an existing robot."""
        robot_data = self._to_db_dict(robot, project_id)
        robot_data.pop("id")  # Don't update the ID
        robot_data.pop("project_id")  # Don't update project_id

        stmt = (
            update(ProjectRobot)
            .where(
                ProjectRobot.project_id == str(project_id),
                ProjectRobot.id == str(robot_id),
            )
            .values(**robot_data)
        )

        await self.db.execute(stmt)
        await self.db.commit()

        # Fetch and return the updated robot
        updated_robot = await self.get_by_id(project_id, robot_id)
        if updated_robot is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

        return updated_robot

    async def delete_by_id(self, project_id: UUID, robot_id: UUID) -> None:
        """Delete a robot by ID."""
        stmt = delete(ProjectRobot).where(
            ProjectRobot.project_id == str(project_id), ProjectRobot.id == str(robot_id)
        )

        await self.db.execute(stmt)
        await self.db.commit()


class RobotService:
    @staticmethod
    async def get_robot_list(project_id: UUID) -> list[Robot]:
        async with get_async_db_session_ctx() as session:
            repo = RobotRepository(session)
            return await repo.get_all(project_id)

    @staticmethod
    async def get_robot_by_id(project_id: UUID, robot_id: UUID) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = RobotRepository(session)
            robot = await repo.get_by_id(project_id, robot_id)

            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(project_id))

            return robot

    @staticmethod
    async def create_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = RobotRepository(session)
            return await repo.save(project_id, robot)

    @staticmethod
    async def update_robot(project_id: UUID, robot: Robot) -> Robot:
        async with get_async_db_session_ctx() as session:
            repo = RobotRepository(session)
            return await repo.update(project_id, robot.id, robot)

    @staticmethod
    async def delete_robot(project_id: UUID, robot_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = RobotRepository(session)

            robot = await repo.get_by_id(project_id, robot_id)
            if robot is None:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))

            await repo.delete_by_id(project_id, robot_id)
