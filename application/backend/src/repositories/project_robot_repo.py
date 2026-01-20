from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ProjectRobotDB
from repositories.base import ProjectBaseRepository
from repositories.mappers import ProjectRobotMapper
from schemas.robot import Robot


class ProjectRobotRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: UUID):
        super().__init__(db, project_id, ProjectRobotDB)

    @property
    def to_schema(self) -> Callable[[Robot], ProjectRobotDB]:
        return ProjectRobotMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectRobotDB], Robot]:
        return ProjectRobotMapper.from_schema
