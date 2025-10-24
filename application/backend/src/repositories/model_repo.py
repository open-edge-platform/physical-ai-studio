from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ModelDB
from repositories.base import BaseRepository
from repositories.mappers import ModelMapper
from schemas import Model


class ModelRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ModelDB)

    @property
    def to_schema(self) -> Callable[[Model], ModelDB]:
        return ModelMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ModelDB], Model]:
        return ModelMapper.from_schema