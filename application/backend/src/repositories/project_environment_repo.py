from collections.abc import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ProjectCameraDB, ProjectEnvironmentDB, ProjectRobotDB
from repositories.base import ProjectBaseRepository
from repositories.mappers import ProjectCameraMapper, ProjectEnvironmentMapper, ProjectRobotMapper
from schemas.environment import (
    Environment,
    EnvironmentWithRelations,
    RobotEnvironmentConfiguration,
    RobotWithTeleoperator,
    TeleoperatorNoneWithRobot,
    TeleoperatorRobot,
    TeleoperatorRobotWithRobot,
)
from schemas.project_camera import Camera
from schemas.robot import Robot


class ProjectEnvironmentRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: UUID):
        super().__init__(db, project_id, ProjectEnvironmentDB)

    @property
    def to_schema(self) -> Callable[[Environment], ProjectEnvironmentDB]:
        return ProjectEnvironmentMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectEnvironmentDB], Environment]:
        return ProjectEnvironmentMapper.from_schema

    async def get_by_id_with_relations(self, environment_id: UUID) -> EnvironmentWithRelations | None:
        """Get an environment by ID with eager loaded robots and cameras."""
        env = await self.get_by_id(environment_id)
        if env is None:
            return None

        robots_map = await self._fetch_robots_map(env.robots)
        cameras = await self._fetch_cameras(env.camera_ids)
        robots_with_teleoperators = self._build_robots_with_teleoperators(env.robots, robots_map)

        return EnvironmentWithRelations(
            id=env.id,
            name=env.name,
            robots=robots_with_teleoperators,
            cameras=cameras,
            created_at=env.created_at,
            updated_at=env.updated_at,
        )

    async def _fetch_robots_map(self, robot_configs: list[RobotEnvironmentConfiguration]) -> dict[str, Robot]:
        """Fetch all robots involved in the environment and return a map by ID."""
        robot_ids: set[str] = set()
        for config in robot_configs:
            robot_ids.add(str(config.robot_id))
            if isinstance(config.tele_operator, TeleoperatorRobot):
                robot_ids.add(str(config.tele_operator.robot_id))

        if not robot_ids:
            return {}

        stmt = select(ProjectRobotDB).where(
            ProjectRobotDB.project_id == self.project_id,
            ProjectRobotDB.id.in_(list(robot_ids)),
        )
        result = await self.db.execute(stmt)
        return {str(db_robot.id): ProjectRobotMapper.from_schema(db_robot) for db_robot in result.scalars().all()}

    async def _fetch_cameras(self, camera_ids: list[UUID]) -> list[Camera]:
        """Fetch all cameras in the environment."""
        if not camera_ids:
            return []

        stmt = select(ProjectCameraDB).where(
            ProjectCameraDB.project_id == self.project_id,
            ProjectCameraDB.id.in_([str(cid) for cid in camera_ids]),
        )
        result = await self.db.execute(stmt)
        return [ProjectCameraMapper.from_schema(db_camera) for db_camera in result.scalars().all()]

    def _build_robots_with_teleoperators(
        self,
        robot_configs: list[RobotEnvironmentConfiguration],
        robots_map: dict[str, Robot],
    ) -> list[RobotWithTeleoperator]:
        """Construct the list of robots with their eager-loaded teleoperators."""
        robots_with_teleoperators = []
        for config in robot_configs:
            robot = robots_map.get(str(config.robot_id))
            if robot is None:
                continue

            if isinstance(config.tele_operator, TeleoperatorRobot):
                tele_robot = robots_map.get(str(config.tele_operator.robot_id))
                tele_operator = TeleoperatorRobotWithRobot(robot_id=config.tele_operator.robot_id, robot=tele_robot)
            else:
                tele_operator = TeleoperatorNoneWithRobot()

            robots_with_teleoperators.append(RobotWithTeleoperator(robot=robot, tele_operator=tele_operator))

        return robots_with_teleoperators
