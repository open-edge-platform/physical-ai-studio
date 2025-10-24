from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ProjectDB
from repositories.base import BaseRepository
from repositories.mappers import ProjectMapper
from schemas import Project


class ProjectRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ProjectDB)

    @property
    def to_schema(self) -> Callable[[Project], ProjectDB]:
        return ProjectMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectDB], Project]:
        return ProjectMapper.from_schema

    async def get_by_name(self, name: str) -> list[Project]:
        return await self.get_all(extra_filters={"name": name})
