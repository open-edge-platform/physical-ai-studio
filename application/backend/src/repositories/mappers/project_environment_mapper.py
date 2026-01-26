import json
from typing import Any
from uuid import UUID

from db.schema import ProjectEnvironmentDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.environment import Environment, RobotEnvironmentConfiguration, TeleoperatorNone, TeleoperatorRobot


class ProjectEnvironmentMapper(IBaseMapper):
    """Mapper for Environment schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(environment: Environment) -> ProjectEnvironmentDB:
        """Convert Environment schema to db model."""
        robots_json = json.dumps(
            [
                {
                    "robot_id": str(robot.robot_id),
                    "tele_operator": (
                        {"type": "robot", "robot_id": str(robot.tele_operator.robot_id)}
                        if isinstance(robot.tele_operator, TeleoperatorRobot)
                        else {"type": "none"}
                    ),
                }
                for robot in environment.robots
            ]
        )

        camera_ids_json = json.dumps([str(camera_id) for camera_id in environment.camera_ids])

        return ProjectEnvironmentDB(
            id=str(environment.id),
            project_id="",  # Will be set by repository
            name=environment.name,
            robots=robots_json,
            camera_ids=camera_ids_json,
            created_at=environment.created_at,
            updated_at=environment.updated_at,
        )

    @staticmethod
    def from_schema(db_env: ProjectEnvironmentDB) -> Environment:
        """Convert Environment db entity to schema."""
        # Parse robots JSON
        robots_data = ProjectEnvironmentMapper._parse_json(db_env.robots, [])
        robots = [
            RobotEnvironmentConfiguration(
                robot_id=UUID(rc["robot_id"]),
                tele_operator=(
                    TeleoperatorRobot(robot_id=UUID(rc["tele_operator"]["robot_id"]))
                    if rc.get("tele_operator", {}).get("type") == "robot"
                    else TeleoperatorNone()
                ),
            )
            for rc in robots_data
        ]

        # Parse camera_ids JSON
        camera_ids_data = ProjectEnvironmentMapper._parse_json(db_env.camera_ids, [])
        camera_ids = [UUID(cid) for cid in camera_ids_data]

        return Environment(
            id=db_env.id,
            name=db_env.name,
            robots=robots,
            camera_ids=camera_ids,
            created_at=db_env.created_at,
            updated_at=db_env.updated_at,
        )

    @staticmethod
    def _parse_json(value: Any, default: Any) -> Any:
        """Safely parse JSON that might be a string or already parsed."""
        if value is None:
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value
