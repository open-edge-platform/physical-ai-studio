from sqlalchemy.orm import Session

from db.schema import ProjectDB
from repositories.base import BaseRepository


class ProjectRepository(BaseRepository[ProjectDB]):
    """Repository for project-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ProjectDB)

    def save(self, project: ProjectDB) -> ProjectDB:
        return super().save(project)