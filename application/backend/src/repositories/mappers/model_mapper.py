from db.schema import ModelDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Model


class ModelMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Model) -> ModelDB:
        return ModelDB(**db_schema.model_dump(mode="json"))

    @staticmethod
    def from_schema(model: ModelDB) -> Model:
        return Model.model_validate(model, from_attributes=True)
