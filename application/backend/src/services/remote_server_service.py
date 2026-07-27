from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from core.secret_encryption import decrypt_secret, encrypt_secret
from db import get_async_db_session_ctx
from exceptions import ResourceAlreadyExistsError, ResourceNotFoundError, ResourceType
from repositories.remote_server_repo import RemoteServerRepository
from schemas.remote_server import (
    LastCheckSummary,
    RemoteServer,
    RemoteServerCreate,
    RemoteServerInternal,
    RemoteServerUpdate,
)

_CONFLICT_DETAIL = "A server with this host, port, and username is already configured."


class RemoteServerService:
    """Manage globally registered SSH-accessible remote server configurations.

    Persists servers with their SSH secrets encrypted at rest. This service only
    owns durable storage and the encryption boundary.
    """

    @staticmethod
    async def list_remote_servers() -> list[RemoteServer]:
        """Return registered servers, sanitized, ordered by their creation time."""
        async with get_async_db_session_ctx() as session:
            records = await RemoteServerRepository(session).list_ordered()
            return [record.to_public() for record in records]

    @staticmethod
    async def get_remote_server(remote_server_id: UUID) -> RemoteServer:
        """Return one registered server, sanitized, or raise a not-found error."""
        record = await RemoteServerService._get_record(remote_server_id)
        return record.to_public()

    @staticmethod
    async def get_remote_server_record(remote_server_id: UUID) -> RemoteServerInternal:
        """Return the full internal record, including encrypted secrets and the host key.

        Reserved for the SSH provisioning boundary. Never return this value from an HTTP response.
        """
        return await RemoteServerService._get_record(remote_server_id)

    @staticmethod
    async def _get_record(remote_server_id: UUID) -> RemoteServerInternal:
        async with get_async_db_session_ctx() as session:
            record = await RemoteServerRepository(session).get_by_id(remote_server_id)
            if record is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
            return record

    @staticmethod
    async def create_remote_server(config: RemoteServerCreate) -> RemoteServer:
        """Persist a new registered SSH remote server with its secrets encrypted."""
        record = RemoteServerInternal(
            id=uuid4(),
            name=config.name,
            host=config.host,
            port=config.port,
            username=config.username,
            auth_type=config.auth_type,
            device_type=config.device_type,
            last_check=LastCheckSummary(),
            ssh_secret_encrypted=encrypt_secret(config.ssh_secret),
            ssh_key_passphrase_encrypted=(
                encrypt_secret(config.ssh_key_passphrase) if config.ssh_key_passphrase is not None else None
            ),
            host_key=None,
        )
        async with get_async_db_session_ctx() as session:
            try:
                saved = await RemoteServerRepository(session).save(record)
            except IntegrityError as error:
                await session.rollback()
                raise ResourceAlreadyExistsError("Remote server", _CONFLICT_DETAIL) from error
        return saved.to_public()

    @staticmethod
    async def update_remote_server(remote_server_id: UUID, update: RemoteServerUpdate) -> RemoteServer:
        """Update a registered server, re-encrypting any rotated secret fields."""
        async with get_async_db_session_ctx() as session:
            repository = RemoteServerRepository(session)
            record = await repository.get_by_id(remote_server_id)
            if record is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))

            partial_update = update.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude={"ssh_secret", "ssh_key_passphrase"},
            )
            if update.ssh_secret is not None:
                partial_update["ssh_secret_encrypted"] = encrypt_secret(update.ssh_secret)
            if update.ssh_key_passphrase is not None:
                partial_update["ssh_key_passphrase_encrypted"] = encrypt_secret(update.ssh_key_passphrase)

            try:
                saved = await repository.update(record, partial_update)
            except IntegrityError as error:
                await session.rollback()
                raise ResourceAlreadyExistsError("Remote server", _CONFLICT_DETAIL) from error
        return saved.to_public()

    @staticmethod
    async def delete_remote_server(remote_server_id: UUID) -> None:
        """Delete a registered server."""
        async with get_async_db_session_ctx() as session:
            repository = RemoteServerRepository(session)
            if await repository.get_by_id(remote_server_id) is None:
                raise ResourceNotFoundError(ResourceType.REMOTE_SERVER, str(remote_server_id))
            await repository.delete_by_id(remote_server_id)

    @staticmethod
    def decrypt_ssh_secret(record: RemoteServerInternal) -> str:
        """Decrypt the stored SSH secret.

        Reserved for the SSH provisioning boundary. Never return this value
        from an HTTP response or write it to logs.
        """
        return decrypt_secret(record.ssh_secret_encrypted)

    @staticmethod
    def decrypt_ssh_key_passphrase(record: RemoteServerInternal) -> str | None:
        """Decrypt the stored SSH key passphrase, if one is configured.

        Reserved for the SSH provisioning boundary. Never return this value
        from an HTTP response or write it to logs.
        """
        if record.ssh_key_passphrase_encrypted is None:
            return None
        return decrypt_secret(record.ssh_key_passphrase_encrypted)
