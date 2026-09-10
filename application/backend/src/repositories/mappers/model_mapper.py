from db.schema import ModelDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Model


class ModelMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Model) -> ModelDB:
        # available_backends and snapflow_enabled are computed fields derived
        # from the filesystem / properties, not their own ModelDB columns.
        return ModelDB(**db_schema.model_dump(mode="json", exclude={"available_backends", "snapflow_enabled"}))

    @staticmethod
    def from_schema(model: ModelDB) -> Model:
        return Model.model_validate(model, from_attributes=True)
