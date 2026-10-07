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
            connection_mode=db_schema.connection.connection_mode.value,
            connection=db_schema.connection.model_dump(mode="json"),
        )

    @staticmethod
    def from_schema(model: RemoteTrainerDB) -> RemoteTrainer:
        """Convert a database model to its API schema."""
        return RemoteTrainer.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "connection": model.connection,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
