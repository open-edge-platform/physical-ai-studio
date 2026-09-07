from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from schemas.project_camera import Camera


class ProjectCameraService:
    def __init__(self, session: AsyncSession, catalog_registry: RobotCatalogRegistry) -> None:
        self.session = session
        self.catalog_registry = catalog_registry

    def _repo(self, project_id: UUID) -> ProjectCameraRepository:
        return ProjectCameraRepository(self.session, str(project_id))

    async def get_camera_list(self, project_id: UUID) -> list[Camera]:
        return await self._repo(project_id).get_all()

    async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> Camera:
        camera = await self._repo(project_id).get_by_id(camera_id)

        if camera is None:
            raise ResourceNotFoundError(ResourceType.CAMERA, str(project_id))

        return camera

    async def create_camera(self, project_id: UUID, camera: Camera) -> Camera:
        return await self._repo(project_id).save(camera)

    async def update_camera(self, project_id: UUID, partial_camera: Camera) -> Camera:
        repo = self._repo(project_id)
        camera = await repo.get_by_id(partial_camera.id)
        if camera is None:
            raise ResourceNotFoundError(ResourceType.CAMERA, str(partial_camera.id))

        return await repo.update(camera, partial_update=partial_camera.model_dump(exclude={"id"}))

    async def delete_camera(self, project_id: UUID, camera_id: UUID) -> None:
        repo = self._repo(project_id)
        camera = await repo.get_by_id(camera_id)
        if camera is None:
            raise ResourceNotFoundError(ResourceType.CAMERA, str(camera_id))

        try:
            await repo.delete_by_id(camera_id)
        except IntegrityError as e:
            await self.session.rollback()
            env_repo = ProjectEnvironmentRepository(self.session, project_id, self.catalog_registry)
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
