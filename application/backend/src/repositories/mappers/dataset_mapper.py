from db.schema import DatasetDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Dataset


class DatasetMapper(IBaseMapper):
    """Mapper for Dataset schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(dataset: Dataset) -> DatasetDB:
        """Convert Dataset schema to db model."""
        return DatasetDB(
            id=str(dataset.id),
            name=dataset.name,
            path=dataset.path,
            project_id=str(dataset.project_id),
            environment_id=str(dataset.environment_id),
        )

    @staticmethod
    def from_schema(dataset_db: DatasetDB) -> Dataset:
        """Convert Dataset db entity to schema."""
        return Dataset.model_validate(dataset_db, from_attributes=True)
