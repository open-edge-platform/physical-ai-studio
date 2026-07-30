from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceNotFoundError, ResourceType
from repositories import ProjectRepository
from schemas import Project


class ProjectService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = ProjectRepository(session)

    async def get_project_list(self) -> list[Project]:
        return await self.repo.get_all()

    async def get_project_by_id(self, project_id: UUID) -> Project:
        project = await self.repo.get_by_id(project_id)

        if project is None:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        return project

    async def create_project(self, project: Project) -> Project:
        return await self.repo.save(project)

    async def update_project(self, project: Project, partial_config: dict) -> Project:
        return await self.repo.update(project, partial_config)

    async def delete_project(self, project_id: UUID) -> None:
        project = await self.repo.get_by_id(project_id)
        if project is None:
            raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))
        await self.repo.delete_by_id(project_id)
