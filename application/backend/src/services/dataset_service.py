from uuid import UUID
from db import get_db_session
from repositories import DatasetRepository
from schemas import Dataset
from services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from services.mappers.datasets_mapper import DatasetMapper
from services.parent_process_guard import parent_process_only


class DatasetService:
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Dataset, DatasetRepository] = GenericPersistenceService(
            ServiceConfig(DatasetRepository, DatasetMapper, ResourceType.DATASET)
        )

    @parent_process_only
    def create_dataset(self, dataset: Dataset) -> Dataset:
        return self._persistence.create(dataset)

