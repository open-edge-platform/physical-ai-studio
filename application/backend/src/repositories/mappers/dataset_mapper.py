from db.schema import DatasetDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Dataset


class DatasetMapper(IBaseMapper):
    @staticmethod
    def to_schema(dataset: Dataset) -> DatasetDB:
        return DatasetDB(**dataset.model_dump(mode="json"))

    @staticmethod
    def from_schema(model_db: DatasetDB) -> Dataset:
        return Dataset.model_validate(model_db, from_attributes=True)