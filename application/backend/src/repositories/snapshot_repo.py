from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import SnapshotDB
from repositories.base import BaseRepository
from repositories.mappers import SnapshotMapper
from schemas import Snapshot


class SnapshotRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, SnapshotDB)

    @property
    def to_schema(self) -> Callable[[Snapshot], SnapshotDB]:
        return SnapshotMapper.to_schema

    @property
    def from_schema(self) -> Callable[[SnapshotDB], Snapshot]:
        return SnapshotMapper.from_schema
