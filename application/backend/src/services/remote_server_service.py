# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Persistence for SSH-provisioned training servers.

Create and update here only persist the row. Tier-1-preflight-gated saves and
resolved-host display are layered on top by the API; this service never
dials SSH.
"""

from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.remote_server_repo import RemoteServerRepository
from schemas.remote_server import RemoteServer, RemoteServerCreate, RemoteServerUpdate


class RemoteServerService:
    """Manage SSH-provisioned training server registrations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = RemoteServerRepository(session)

    async def list_remote_servers(self) -> list[RemoteServer]:
        """Return registered servers in stable creation order."""
        return await self.repo.list_ordered()

    async def get_remote_server(self, remote_server_id: UUID) -> RemoteServer:
        """Return one registered server or raise a not-found error."""
        remote_server = await self.repo.get_by_id(remote_server_id)
        if remote_server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        return remote_server

    async def create_remote_server(self, config: RemoteServerCreate) -> RemoteServer:
        """Persist an SSH-provisioned training server."""
        remote_server = RemoteServer(id=uuid4(), **config.model_dump())
        try:
            return await self.repo.save(remote_server)
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote server",
                "A server with this SSH host alias is already configured.",
            ) from error

    async def update_remote_server(self, remote_server_id: UUID, update: RemoteServerUpdate) -> RemoteServer:
        """Update a registered server's mutable fields."""
        remote_server = await self.repo.get_by_id(remote_server_id)
        if remote_server is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        try:
            return await self.repo.update(remote_server, update.model_dump(exclude_none=True, exclude_unset=True))
        except IntegrityError as error:
            await self.session.rollback()
            raise ResourceAlreadyExistsError(
                "Remote server",
                "A server with this SSH host alias is already configured.",
            ) from error

    async def delete_remote_server(self, remote_server_id: UUID) -> None:
        """Delete a registered server."""
        if await self.repo.get_by_id(remote_server_id) is None:
            raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
        await self.repo.delete_by_id(remote_server_id)
