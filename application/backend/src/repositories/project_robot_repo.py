from collections.abc import Callable
from uuid import UUID

from sqlalchemy import delete
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

    # A simulation start saves a robot and its re-pointed leaders in one transaction, and deleting
    # a robot deletes its simulation cameras in the same one. These only flush: the caller commits.

    async def stage_update(self, robot: Robot) -> None:
        """Save *robot* as it is, without committing."""
        schema_item = self.to_schema(robot)
        schema_item.project_id = self.project_id
        await self.db.merge(schema_item)
        await self.db.flush()

    async def stage_delete(self, robot_id: UUID) -> None:
        """Delete the robot, without committing."""
        await self.db.execute(
            delete(ProjectRobotDB).where(
                ProjectRobotDB.id == str(robot_id), ProjectRobotDB.project_id == self.project_id
            )
        )
