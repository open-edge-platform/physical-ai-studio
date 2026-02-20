from db.schema import DatasetDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Dataset


class DatasetMapper(IBaseMapper):
    """Mapper for Dataset schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Dataset) -> DatasetDB:
        """Convert Dataset schema to db model."""
        return DatasetDB(
            id=str(db_schema.id),
            name=db_schema.name,
            path=db_schema.path,
            project_id=str(db_schema.project_id),
            environment_id=str(db_schema.environment_id),
        )

    @staticmethod
    def from_schema(model: DatasetDB) -> Dataset:
        """Convert Dataset db entity to schema."""
        return Dataset.model_validate(model, from_attributes=True)
