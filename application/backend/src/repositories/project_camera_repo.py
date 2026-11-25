from collections.abc import Callable
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import EnvironmentCameraDB, ProjectCameraDB
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

    async def add_if_absent(self, camera: Camera) -> None:
        """Insert *camera* unless one with its id exists, without committing."""
        row = self.to_schema(camera)
        row.project_id = self.project_id
        values = {
            column.key: getattr(row, column.key)
            for column in ProjectCameraDB.__table__.columns
            if getattr(row, column.key) is not None
        }
        await self.db.execute(insert(ProjectCameraDB).values(**values).on_conflict_do_nothing())

    async def delete_simulation_cameras(self, robot_id: UUID) -> None:
        """Delete the robot's simulation cameras and their environment links, without committing.

        Without the robot they have no simulation to stream from. The caller commits, with the
        robot's own deletion.
        """
        rows = (
            await self.db.execute(
                select(ProjectCameraDB.id, ProjectCameraDB.payload).where(
                    ProjectCameraDB.project_id == self.project_id, ProjectCameraDB.driver == "simulation"
                )
            )
        ).all()
        camera_ids = [
            row.id for row in rows if isinstance(row.payload, dict) and row.payload.get("robot_id") == str(robot_id)
        ]
        if not camera_ids:
            return
        await self.db.execute(delete(EnvironmentCameraDB).where(EnvironmentCameraDB.camera_id.in_(camera_ids)))
        await self.db.execute(delete(ProjectCameraDB).where(ProjectCameraDB.id.in_(camera_ids)))
