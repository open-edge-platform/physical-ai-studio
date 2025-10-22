from uuid import UUID

from repositories import DatasetRepository
from schemas import Model
from services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from services.mappers import DatasetMapper
from services.parent_process_guard import parent_process_only


class ModelService:
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Model, DatasetRepository] = GenericPersistenceService(
            ServiceConfig(DatasetRepository, DatasetMapper, ResourceType.DATASET)
        )

    def list_models(self) -> list[Model]:
        return self._persistence.list_all()

    @parent_process_only
    def create_model(self, model: Model) -> Model:
        return self._persistence.create(model)

    def get_model_by_id(self, model_id: UUID) -> Model:
        model = self._persistence.get_by_id(model_id)
        if not model:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return model

