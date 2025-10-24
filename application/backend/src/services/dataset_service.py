from uuid import UUID

from db import get_async_db_session_ctx
from repositories import DatasetRepository
from schemas import Dataset


class DatasetService:
    @staticmethod
    async def get_dataset_list() -> list[Dataset]:
        async with get_async_db_session_ctx() as session:
            repo = DatasetRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_dataset_by_id(dataset_id: UUID) -> Dataset | None:
        async with get_async_db_session_ctx() as session:
            repo = DatasetRepository(session)
            return await repo.get_by_id(dataset_id)

    @staticmethod
    async def create_dataset(dataset: Dataset) -> Dataset:
        async with get_async_db_session_ctx() as session:
            repo = DatasetRepository(session)
            return await repo.save(dataset)

    @staticmethod
    async def delete_dataset(dataset_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = DatasetRepository(session)
            await repo.delete_by_id(dataset_id)
