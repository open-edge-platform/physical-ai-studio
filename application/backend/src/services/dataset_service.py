from uuid import UUID

from db import get_db_session
from repositories import DatasetRepository
from schemas import Dataset
from services.base import GenericPersistenceService, ResourceNotFoundError, ResourceType, ServiceConfig
from services.mappers import DatasetMapper
from services.parent_process_guard import parent_process_only


class DatasetService:
    def __init__(self) -> None:
        self._persistence: GenericPersistenceService[Dataset, DatasetRepository] = GenericPersistenceService(
            ServiceConfig(DatasetRepository, DatasetMapper, ResourceType.DATASET)
        )

    @parent_process_only
    def create_dataset(self, dataset: Dataset) -> Dataset:
        return self._persistence.create(dataset)

    def get_dataset_by_id(self, dataset_id: UUID) -> Dataset:
        dataset = self._persistence.get_by_id(dataset_id)
        if not dataset:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))
        return dataset

    @parent_process_only
    def delete_dataset_by_id(self, dataset_id: UUID) -> None:
        with get_db_session() as db:
            self._persistence.delete_by_id(dataset_id, db)
            db.commit()
