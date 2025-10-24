from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import DatasetDB
from repositories.base import BaseRepository
from repositories.mappers import DatasetMapper
from schemas import Dataset


class DatasetRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, DatasetDB)

    @property
    def to_schema(self) -> Callable[[Dataset], DatasetDB]:
        return DatasetMapper.to_schema

    @property
    def from_schema(self) -> Callable[[DatasetDB], Dataset]:
        return DatasetMapper.from_schema