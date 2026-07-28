from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_environment_repo import ProjectEnvironmentRepository
from schemas.environment import Environment, EnvironmentWithRelations


class EnvironmentService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _repo(self, project_id: UUID) -> ProjectEnvironmentRepository:
        return ProjectEnvironmentRepository(self.session, project_id)

    async def get_environment_list(self, project_id: UUID) -> list[Environment]:
        return await self._repo(project_id).get_all()

    async def get_environment_by_id(self, project_id: UUID, environment_id: UUID) -> EnvironmentWithRelations:
        environment = await self._repo(project_id).get_by_id_with_relations(environment_id)

        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        return environment

    async def create_environment(self, project_id: UUID, environment: Environment) -> Environment:
        return await self._repo(project_id).save(environment)

    async def update_environment(self, project_id: UUID, environment: Environment) -> EnvironmentWithRelations:
        repo = self._repo(project_id)
        existing = await repo.get_by_id(environment.id)
        if existing is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

        await repo.update(existing, environment.model_dump(exclude={"id", "created_at", "updated_at"}))

        updated = await repo.get_by_id_with_relations(environment.id)
        if updated is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

        return updated

    async def delete_environment(self, project_id: UUID, environment_id: UUID) -> None:
        repo = self._repo(project_id)

        environment = await repo.get_by_id(environment_id)
        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        await repo.delete_by_id(environment_id)
