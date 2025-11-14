from pathlib import Path
from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories import ModelRepository
from schemas import Model


class ModelService:
    @staticmethod
    async def get_model_list() -> list[Model]:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_model_by_id(model_id: UUID) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            model = await repo.get_by_id(model_id)
            if model is None:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

            return model

    @staticmethod
    async def create_model(model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.save(model)

    @staticmethod
    async def update_model(model: Model, update: dict) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.update(model, update)

    @staticmethod
    async def delete_model(model: Model) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            await repo.delete_by_id(model.id)
            model_path = Path(model.path).expanduser()
            model_path.unlink()
            model_path.parent.rmdir()

    @staticmethod
    async def get_project_models(project_id: UUID) -> list[Model]:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session)
            return await repo.get_project_models(project_id)
