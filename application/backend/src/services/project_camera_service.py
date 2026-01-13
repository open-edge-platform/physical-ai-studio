from uuid import UUID

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.project_camera_repo import ProjectCameraRepository
from schemas.project_camera import Camera


class ProjectCameraService:
    @staticmethod
    async def get_camera_list(project_id: UUID) -> list[Camera]:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))
            return await repo.get_all()

    @staticmethod
    async def get_camera_by_id(project_id: UUID, camera_id: UUID) -> Camera:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))
            camera = await repo.get_by_id(camera_id)

            if camera is None:
                raise ResourceNotFoundError(ResourceType.CAMERA, str(project_id))

            return camera

    @staticmethod
    async def create_camera(project_id: UUID, camera: Camera) -> Camera:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))
            return await repo.save(camera)

    @staticmethod
    async def update_camera(project_id: UUID, camera: Camera) -> Camera:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))
            return await repo.update(camera, partial_update=camera.model_dump(exclude={"id"}))

    @staticmethod
    async def delete_camera(project_id: UUID, camera_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))

            camera = await repo.get_by_id(camera_id)
            if camera is None:
                raise ResourceNotFoundError(ResourceType.CAMERA, str(camera_id))

            await repo.delete_by_id(camera_id)
