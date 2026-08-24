from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ProjectRobotDB
from repositories.base import ProjectBaseRepository
from repositories.mappers import ProjectRobotMapper
from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import Robot, UnavailableRobot


class ProjectRobotRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: UUID, catalog_registry: RobotCatalogRegistry):
        super().__init__(db, project_id, ProjectRobotDB)
        self.catalog_registry = catalog_registry

    @property
    def to_schema(self) -> Callable[[Robot], ProjectRobotDB]:
        return ProjectRobotMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectRobotDB], Robot | UnavailableRobot]:
        return lambda model: ProjectRobotMapper.from_schema(model, self.catalog_registry)
