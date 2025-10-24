from uuid import UUID

from db import get_async_db_session_ctx
from repositories import ProjectRepository
from schemas import Project


class ProjectService:
    @staticmethod
    async def get_project_list() -> list[Project]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.get_all()

    @staticmethod
    async def get_project_by_id(project_id: UUID) -> Project | None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.get_by_id(project_id)

    @staticmethod
    async def create_project(project: Project) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.save(project)

    @staticmethod
    async def delete_project(project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            await repo.delete_by_id(project_id)