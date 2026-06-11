from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import EnvironmentRobotDB, ProjectEnvironmentDB
from repositories.base import ProjectBaseRepository
from repositories.mappers import ProjectCameraMapper, ProjectEnvironmentMapper, ProjectRobotMapper
from schemas.environment import (
    Environment,
    EnvironmentWithRelations,
    RobotWithTeleoperator,
    TeleoperatorNoneWithRobot,
    TeleoperatorRobotWithRobot,
)


class ProjectEnvironmentRepository(ProjectBaseRepository[Environment, ProjectEnvironmentDB]):
    def __init__(self, db: AsyncSession, project_id: UUID):
        super().__init__(db, project_id, ProjectEnvironmentDB)

    @property
    def to_schema(self) -> Callable[[Environment], ProjectEnvironmentDB]:
        return ProjectEnvironmentMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectEnvironmentDB], Environment]:
        return ProjectEnvironmentMapper.from_schema

    async def update(self, item: Environment, partial_update: dict) -> Environment:
        """Update an environment, fully replacing its robot/camera join rows.

        Reassigning the ``robot_links`` / ``camera_links`` collections lets the ``delete-orphan``
        cascade remove the previous rows and insert the new ones on flush. This is more deterministic
        than ``db.merge`` for the composite-primary-key join tables.
        """
        partial_update = {
            k: v for k, v in partial_update.items() if v is not None and k not in {"created_at", "updated_at"}
        }
        to_update = item.model_copy(update=partial_update, deep=True)
        to_update = item.__class__.model_validate(to_update.model_dump())

        stmt = select(ProjectEnvironmentDB).where(
            ProjectEnvironmentDB.id == str(item.id),
            ProjectEnvironmentDB.project_id == self.project_id,
        )
        existing = (await self.db.execute(stmt)).scalars().first()
        if existing is None:
            raise ValueError(f"{item.__class__} with ID `{item.id}` doesn't exist")

        existing.name = to_update.name
        existing.robot_links = ProjectEnvironmentMapper.build_robot_links(to_update)
        existing.camera_links = ProjectEnvironmentMapper.build_camera_links(to_update)
        await self.db.commit()

        updated = await self.get_by_id(item.id)
        if updated is None:
            raise ValueError(f"{item.__class__} with ID `{item.id}` doesn't exist")
        return updated

    async def get_by_id_with_relations(self, environment_id: UUID) -> EnvironmentWithRelations | None:
        """Get an environment by ID with eager loaded robots and cameras."""
        stmt = select(ProjectEnvironmentDB).where(
            ProjectEnvironmentDB.id == str(environment_id),
            ProjectEnvironmentDB.project_id == self.project_id,
        )
        result = await self.db.execute(stmt)
        env = result.scalars().first()
        if env is None:
            return None

        cameras = [ProjectCameraMapper.from_schema(link.camera) for link in env.camera_links]

        return EnvironmentWithRelations(
            id=env.id,
            name=env.name,
            robots=self._build_robots_with_teleoperators(env.robot_links),
            cameras=cameras,
            created_at=env.created_at,
            updated_at=env.updated_at,
        )

    @staticmethod
    def _build_robots_with_teleoperators(robot_links: list[EnvironmentRobotDB]) -> list[RobotWithTeleoperator]:
        """Construct the list of robots with their eager-loaded teleoperators."""
        robots_with_teleoperators = []
        for link in robot_links:
            robot = ProjectRobotMapper.from_schema(link.robot)

            if link.tele_operator_type == "robot" and link.tele_operator_robot is not None:
                tele_operator = TeleoperatorRobotWithRobot(
                    robot_id=UUID(link.tele_operator_robot_id),
                    robot=ProjectRobotMapper.from_schema(link.tele_operator_robot),
                )
            else:
                tele_operator = TeleoperatorNoneWithRobot()

            robots_with_teleoperators.append(RobotWithTeleoperator(robot=robot, tele_operator=tele_operator))

        return robots_with_teleoperators
