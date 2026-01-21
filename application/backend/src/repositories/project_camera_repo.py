from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ProjectCameraDB
from repositories.base import ProjectBaseRepository
from repositories.mappers import ProjectCameraMapper
from schemas.project_camera import Camera


class ProjectCameraRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: str):
        super().__init__(db, project_id, ProjectCameraDB)

    @property
    def to_schema(self) -> Callable[[Camera], ProjectCameraDB]:
        return ProjectCameraMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectCameraDB], Camera]:
        return ProjectCameraMapper.from_schema
