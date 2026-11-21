from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import DatasetDB
from exceptions import ResourceInUseError, ResourceNotFoundError, ResourceType, SimulationNotSupportedError
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.environment import Environment, EnvironmentWithRelations


class EnvironmentService:
    def __init__(self, session: AsyncSession, catalog_registry: RobotCatalogRegistry) -> None:
        self.session = session
        self.catalog_registry = catalog_registry

    def _repo(self, project_id: UUID) -> ProjectEnvironmentRepository:
        return ProjectEnvironmentRepository(self.session, project_id, self.catalog_registry)

    async def get_environment_list(self, project_id: UUID) -> list[Environment]:
        return await self._repo(project_id).get_all()

    async def get_environment_by_id(self, project_id: UUID, environment_id: UUID) -> EnvironmentWithRelations:
        environment = await self._repo(project_id).get_by_id_with_relations(environment_id)

        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        return environment

    async def create_environment(self, project_id: UUID, environment: Environment) -> Environment:
        await self._check_autopilots(project_id, environment)
        return await self._repo(project_id).save(environment)

    async def update_environment(self, project_id: UUID, environment: Environment) -> EnvironmentWithRelations:
        repo = self._repo(project_id)
        existing = await repo.get_by_id(environment.id)
        if existing is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))
        await self._check_autopilots(project_id, environment)

        await repo.update(existing, environment.model_dump(exclude={"id", "created_at", "updated_at"}))

        updated = await repo.get_by_id_with_relations(environment.id)
        if updated is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment.id))

        return updated

    async def _check_autopilots(self, project_id: UUID, environment: Environment) -> None:
        """Only a simulated robot can be led by its simulation's autopilot.

        Raises:
            ResourceNotFoundError: A robot led by the autopilot does not exist.
            SimulationNotSupportedError: A robot led by the autopilot has no simulation.
        """
        robots = RobotService(self.session, self.catalog_registry)
        for configuration in environment.robots:
            if configuration.tele_operator.type != "autopilot":
                continue
            robot = await robots.get_robot_by_id(project_id, configuration.robot_id)
            if getattr(self.catalog_registry.get_definition(robot.type), "simulation", None) is None:
                raise SimulationNotSupportedError(
                    f"Robot {robot.name!r} ({robot.type}) has no simulation, so no autopilot can lead it."
                )

    async def delete_environment(self, project_id: UUID, environment_id: UUID) -> None:
        repo = self._repo(project_id)

        environment = await repo.get_by_id(environment_id)
        if environment is None:
            raise ResourceNotFoundError(ResourceType.ENVIRONMENT, str(environment_id))

        try:
            await repo.delete_by_id(environment_id)
        except IntegrityError as error:
            await self.session.rollback()
            dataset_names = (
                await self.session.scalars(
                    select(DatasetDB.name).where(
                        DatasetDB.project_id == str(project_id), DatasetDB.environment_id == str(environment_id)
                    )
                )
            ).all()
            if dataset_names:
                raise ResourceInUseError(
                    ResourceType.ENVIRONMENT,
                    environment_id,
                    message=(
                        f"Environment '{environment.name}' cannot be deleted because it is used by dataset(s): "
                        f"{', '.join(dataset_names)}. Delete those datasets first."
                    ),
                ) from error
            raise
