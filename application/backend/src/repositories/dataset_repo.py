from sqlalchemy.orm import Session

from db.schema import DatasetDB
from repositories.base import BaseRepository

class DatasetRepository(BaseRepository[DatasetDB]):
    """Repository for project-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, DatasetDB)

    def save(self, dataset: DatasetDB) -> DatasetDB:
        return super().save(dataset)
