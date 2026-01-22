from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_environment_repo import ProjectEnvironmentRepository
from schemas.environment import Environment, EnvironmentWithRelations


class EnvironmentService:
    @staticmethod
    async def get_environment_list(project_id: UUID) -> list[Environment]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectEnvironmentRepository(session, project_id)
            return await repo.get_all()

    @staticmethod
    async def get_environment_by_id(project_id: UUID, environment_id: UUID) -> EnvironmentWithRelations:
        async with get_async_db_session_ctx() as session:
            repo = ProjectEnvironmentRepository(session, project_id)
            environment = await repo.get_by_id_with_relations(environment_id)

            if environment is None:
                raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

            return environment

    @staticmethod
    async def create_environment(project_id: UUID, environment: Environment) -> Environment:
        async with get_async_db_session_ctx() as session:
            repo = ProjectEnvironmentRepository(session, project_id)
            return await repo.save(environment)

    @staticmethod
    async def update_environment(project_id: UUID, environment: Environment) -> EnvironmentWithRelations:
        async with get_async_db_session_ctx() as session:
            repo = ProjectEnvironmentRepository(session, project_id)

            # Use base repository's update with partial_update dict
            # Then fetch the updated environment with relations
            existing = await repo.get_by_id(environment.id)
            if existing is None:
                raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

            await repo.update(existing, environment.model_dump(exclude={"id", "created_at", "updated_at"}))

            updated = await repo.get_by_id_with_relations(environment.id)
            if updated is None:
                raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

            return updated

    @staticmethod
    async def delete_environment(project_id: UUID, environment_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectEnvironmentRepository(session, project_id)

            environment = await repo.get_by_id(environment_id)
            if environment is None:
                raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

            await repo.delete_by_id(environment_id)
