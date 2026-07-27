from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import RemoteServerDB
from repositories.base import BaseRepository
from repositories.mappers.remote_server_mapper import RemoteServerMapper
from schemas.remote_server import RemoteServerInternal


class RemoteServerRepository(BaseRepository[RemoteServerInternal, RemoteServerDB]):
    """Persistence for registered SSH-accessible remote servers.

    Works with :class:`RemoteServerInternal`, the confidential/internal
    record. Callers must convert to :class:`schemas.remote_server.RemoteServer`
    via ``.to_public()`` before returning results outside the service layer.
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db, RemoteServerDB)

    @property
    def to_schema(self) -> Callable[[RemoteServerInternal], RemoteServerDB]:
        return RemoteServerMapper.to_schema

    @property
    def from_schema(self) -> Callable[[RemoteServerDB], RemoteServerInternal]:
        return RemoteServerMapper.from_schema

    async def list_ordered(self) -> list[RemoteServerInternal]:
        """Return registered servers in stable creation order."""
        query = select(RemoteServerDB).order_by(RemoteServerDB.created_at.asc(), RemoteServerDB.name.asc())
        results = await self.db.execute(query)
        return [self.from_schema(model) for model in results.scalars().all()]
