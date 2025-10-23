from db.schema import ModelDB
from schemas import Model


class ModelMapper:
    """Mapper for Model schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(dataset_db: ModelDB | None) -> Model | None:
        """Convert Model db entity to schema."""
        if dataset_db is None:
            return None

        return Model.model_validate(dataset_db, from_attributes=True)

    @staticmethod
    def from_schema(dataset: Model) -> ModelDB:
        """Convert Model schema to db model."""

        return ModelDB(
            id=str(dataset.id),
            name=dataset.name,
            path=dataset.path,
            properties=dataset.properties,
            project_id=str(dataset.project_id),
        )

