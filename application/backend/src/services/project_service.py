from uuid import UUID

from db import get_async_db_session_ctx
from repositories import ProjectRepository
from schemas import Project
from services.base import ResourceNotFoundError, ResourceType


class ProjectService:
    @staticmethod
    async def get_project_list() -> list[Project]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_project_by_id(project_id: UUID) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            project = await repo.get_by_id(project_id)

            if project is None:
                raise ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

            return project

    @staticmethod
    async def create_project(project: Project) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.save(project)

    @staticmethod
    async def update_project(project: Project, partial_config: dict) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.update(project, partial_config)

    @staticmethod
    async def delete_project(project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            await repo.delete_by_id(project_id)
