from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
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
    async def update_camera(project_id: UUID, partial_camera: Camera) -> Camera:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))

            camera = await repo.get_by_id(partial_camera.id)
            if camera is None:
                raise ResourceNotFoundError(ResourceType.CAMERA, str(partial_camera.id))

            return await repo.update(camera, partial_update=partial_camera.model_dump(exclude={"id"}))

    @staticmethod
    async def delete_camera(project_id: UUID, camera_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectCameraRepository(session, str(project_id))

            camera = await repo.get_by_id(camera_id)
            if camera is None:
                raise ResourceNotFoundError(ResourceType.CAMERA, str(camera_id))

            try:
                await repo.delete_by_id(camera_id)
            except IntegrityError as e:
                await session.rollback()  # clear the failed transaction before reusing the session
                env_repo = ProjectEnvironmentRepository(session, project_id)
                environment_names = await env_repo.find_environment_names_using_camera(camera_id)
                if environment_names:
                    raise ResourceInUseError(
                        ResourceType.CAMERA,
                        str(camera_id),
                        message=(
                            f"Camera '{camera.name}' cannot be deleted because it is used in environment(s): "
                            f"{', '.join(environment_names)}. Remove it from those environments first."
                        ),
                    ) from e
                raise
