from db.schema import RemoteTrainerDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.remote_trainer import ManualSshConnection, RemoteTrainer, RemoteTrainerConnectionMode


class RemoteTrainerMapper(IBaseMapper):
    """Map persisted remote trainer endpoints to API schemas."""

    @staticmethod
    def to_schema(db_schema: RemoteTrainer) -> RemoteTrainerDB:
        """Convert an API schema to its database model."""
        return RemoteTrainerDB(
            id=str(db_schema.id),
            name=db_schema.name,
            connection_mode=db_schema.connection_mode,
            url=str(db_schema.url),
            ssh_host_alias=db_schema.ssh_host_alias,
            ssh_hostname=db_schema.ssh_connection.hostname if db_schema.ssh_connection else None,
            ssh_port=db_schema.ssh_connection.port if db_schema.ssh_connection else None,
            ssh_username=db_schema.ssh_connection.user if db_schema.ssh_connection else None,
            ssh_identity_file=db_schema.ssh_connection.identity_file if db_schema.ssh_connection else None,
            ssh_remote_port=db_schema.ssh_remote_port,
            ssh_local_port=db_schema.ssh_local_port,
        )

    @staticmethod
    def from_schema(model: RemoteTrainerDB) -> RemoteTrainer:
        """Convert a database model to its API schema."""
        ssh_connection = None
        if model.connection_mode == RemoteTrainerConnectionMode.SSH and model.ssh_host_alias is None:
            if model.ssh_hostname is None:
                raise ValueError("Manual SSH trainer is missing ssh_hostname")
            ssh_connection = ManualSshConnection(
                hostname=model.ssh_hostname,
                port=model.ssh_port or 22,
                user=model.ssh_username,
                identity_file=model.ssh_identity_file,
            )

        return RemoteTrainer.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "connection_mode": model.connection_mode,
                "url": model.url,
                "ssh_host_alias": model.ssh_host_alias,
                "ssh_connection": ssh_connection,
                "ssh_remote_port": model.ssh_remote_port,
                "ssh_local_port": model.ssh_local_port,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
