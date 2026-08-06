from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import RemoteServerDB
from repositories.base import BaseRepository
from repositories.mappers.remote_server_mapper import RemoteServerMapper
from schemas.remote_server import RemoteServer


class RemoteServerRepository(BaseRepository[RemoteServer, RemoteServerDB]):
    """Persistence for SSH-provisioned training servers."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, RemoteServerDB)

    @property
    def to_schema(self) -> Callable[[RemoteServer], RemoteServerDB]:
        return RemoteServerMapper.to_schema

    @property
    def from_schema(self) -> Callable[[RemoteServerDB], RemoteServer]:
        return RemoteServerMapper.from_schema

    async def list_ordered(self) -> list[RemoteServer]:
        """Return servers in stable creation order."""
        query = select(RemoteServerDB).order_by(RemoteServerDB.created_at.asc(), RemoteServerDB.name.asc())
        results = await self.db.execute(query)
        return [self.from_schema(model) for model in results.scalars().all()]

    async def get_by_alias(self, alias: str) -> RemoteServer | None:
        """Return the server registered for an SSH host alias, if any.

        The alias is unique, so this identifies at most one server. Used to turn a
        duplicate registration into a clear conflict instead of a database error.
        """
        return await self.get_one(extra_filters={"ssh_host_alias": alias})
