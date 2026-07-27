import shutil
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceNotFoundError, ResourceType
from repositories import DatasetRepository
from schemas import Dataset


class DatasetService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = DatasetRepository(session)

    async def get_dataset_list(self) -> list[Dataset]:
        return await self.repo.get_all()

    async def get_dataset_by_id(self, dataset_id: UUID) -> Dataset:
        dataset = await self.repo.get_by_id(dataset_id)
        if dataset is None:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))
        return dataset

    async def create_dataset(self, dataset: Dataset) -> Dataset:
        return await self.repo.save(dataset)

    async def update_dataset_name(self, dataset_id: UUID, name: str) -> Dataset:
        dataset = await self.repo.get_by_id(dataset_id)
        if dataset is None:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))

        return await self.repo.update(dataset, {"name": name})

    async def delete_dataset(self, dataset_id: UUID, remove_files: bool = False) -> None:
        dataset = await self.repo.get_by_id(dataset_id)
        if dataset is None:
            raise ResourceNotFoundError(ResourceType.DATASET, str(dataset_id))

        await self.repo.delete_by_id(dataset_id)

        if remove_files:
            shutil.rmtree(Path(dataset.path).expanduser())
