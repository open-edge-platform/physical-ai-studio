from sqlalchemy.orm import Session

from db.schema import ModelDB
from repositories.base import BaseRepository


class ModelRepository(BaseRepository[ModelDB]):
    """Repository for model-related database operations."""

    def __init__(self, db: Session):
        super().__init__(db, ModelDB)

    def save(self, dataset: ModelDB) -> ModelDB:
        return super().save(dataset)

