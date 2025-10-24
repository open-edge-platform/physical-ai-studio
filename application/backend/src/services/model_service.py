from uuid import UUID

from db import get_async_db_session_ctx
from repositories import ModelRepository
from schemas import Model


class ModelService:
    @staticmethod
    async def get_model_list() -> list[Model]:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_model_by_id(model_id: UUID) -> Model | None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_by_id(model_id)

    @staticmethod
    async def create_model(model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.save(model)

    @staticmethod
    async def delete_model(model_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            await repo.delete_by_id(model_id)
