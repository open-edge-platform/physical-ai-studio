from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import RemoteTrainerDB
from repositories.base import BaseRepository
from repositories.mappers.remote_trainer_mapper import RemoteTrainerMapper
from schemas.remote_trainer import RemoteTrainer


class RemoteTrainerRepository(BaseRepository[RemoteTrainer, RemoteTrainerDB]):
    """Persistence for configured direct trainer endpoints."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, RemoteTrainerDB)

    @property
    def to_schema(self) -> Callable[[RemoteTrainer], RemoteTrainerDB]:
        return RemoteTrainerMapper.to_schema

    @property
    def from_schema(self) -> Callable[[RemoteTrainerDB], RemoteTrainer]:
        return RemoteTrainerMapper.from_schema

    async def list_ordered(self) -> list[RemoteTrainer]:
        """Return endpoints in stable creation order."""
        query = select(RemoteTrainerDB).order_by(RemoteTrainerDB.created_at.asc(), RemoteTrainerDB.name.asc())
        results = await self.db.execute(query)
        return [self.from_schema(model) for model in results.scalars().all()]
