from db.schema import RemoteTrainerDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.remote_trainer import RemoteTrainer


class RemoteTrainerMapper(IBaseMapper):
    """Map persisted remote trainer endpoints to API schemas."""

    @staticmethod
    def to_schema(db_schema: RemoteTrainer) -> RemoteTrainerDB:
        """Convert an API schema to its database model."""
        return RemoteTrainerDB(
            id=str(db_schema.id),
            name=db_schema.name,
            url=str(db_schema.url),
            ssh_host_alias=db_schema.ssh_host_alias,
            ssh_remote_port=db_schema.ssh_remote_port,
            ssh_local_port=db_schema.ssh_local_port,
        )

    @staticmethod
    def from_schema(model: RemoteTrainerDB) -> RemoteTrainer:
        """Convert a database model to its API schema."""
        return RemoteTrainer.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "url": model.url,
                "ssh_host_alias": model.ssh_host_alias,
                "ssh_remote_port": model.ssh_remote_port,
                "ssh_local_port": model.ssh_local_port,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
