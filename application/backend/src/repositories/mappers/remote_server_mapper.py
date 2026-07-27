from db.schema import RemoteServerDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.remote_server import RemoteServerInternal


class RemoteServerMapper(IBaseMapper):
    """Map persisted remote server records to their internal (secret-bearing) schema."""

    @staticmethod
    def to_schema(db_schema: RemoteServerInternal) -> RemoteServerDB:
        """Convert the internal schema to its database model."""
        return RemoteServerDB(
            id=str(db_schema.id),
            name=db_schema.name,
            host=db_schema.host,
            port=db_schema.port,
            username=db_schema.username,
            auth_type=db_schema.auth_type,
            device_type=db_schema.device_type,
            ssh_secret_encrypted=db_schema.ssh_secret_encrypted,
            ssh_key_passphrase_encrypted=db_schema.ssh_key_passphrase_encrypted,
            host_key=db_schema.host_key,
            last_check_status=db_schema.last_check.status,
            last_check_at=db_schema.last_check.checked_at,
            last_check_latency_ms=db_schema.last_check.latency_ms,
            last_check_reason_code=db_schema.last_check.reason_code,
        )

    @staticmethod
    def from_schema(model: RemoteServerDB) -> RemoteServerInternal:
        """Convert a database model to the internal (secret-bearing) schema."""
        return RemoteServerInternal.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "host": model.host,
                "port": model.port,
                "username": model.username,
                "auth_type": model.auth_type,
                "device_type": model.device_type,
                "ssh_secret_encrypted": model.ssh_secret_encrypted,
                "ssh_key_passphrase_encrypted": model.ssh_key_passphrase_encrypted,
                "host_key": model.host_key,
                "last_check": {
                    "status": model.last_check_status,
                    "checked_at": model.last_check_at,
                    "latency_ms": model.last_check_latency_ms,
                    "reason_code": model.last_check_reason_code,
                },
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
