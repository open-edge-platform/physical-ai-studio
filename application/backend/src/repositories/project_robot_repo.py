from collections.abc import Callable
from typing import Any
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

    # A simulation start saves the robot's new address and its re-pointed leaders in one
    # transaction, and deleting a robot deletes its simulation cameras in the same one: with
    # ``commit=False`` these only flush, and the caller commits.

    async def update(self, item: Any, partial_update: dict, *, commit: bool = True) -> Any:
        if commit:
            return await super().update(item, partial_update)
        partial_update = {
            k: v for k, v in partial_update.items() if v is not None and k not in {"created_at", "updated_at"}
        }
        to_update = item.__class__.model_validate(dict(item.model_copy(update=partial_update, deep=True)))
        schema_item = self.to_schema(to_update)
        schema_item.project_id = self.project_id
        await self.db.merge(schema_item)
        await self.db.flush()
        updated = await self.get_by_id(str(item.id))
        if updated is None:
            raise ValueError(f"{item.__class__} with ID `{item.id}` doesn't exist")
        return updated

    async def delete_by_id(self, obj_id: str | UUID, *, commit: bool = True) -> None:
        if commit:
            return await super().delete_by_id(obj_id)
        await self.db.execute(
            delete(ProjectRobotDB).where(ProjectRobotDB.id == str(obj_id), ProjectRobotDB.project_id == self.project_id)
        )
        return None
