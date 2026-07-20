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
        )

    @staticmethod
    def from_schema(model: RemoteTrainerDB) -> RemoteTrainer:
        """Convert a database model to its API schema."""
        return RemoteTrainer.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "url": model.url,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )
