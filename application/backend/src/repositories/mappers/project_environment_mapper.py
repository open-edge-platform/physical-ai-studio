import json
from typing import Any
from uuid import UUID

from db.schema import ProjectEnvironmentDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.environment import Environment, RobotEnvironmentConfiguration, TeleoperatorNone, TeleoperatorRobot


class ProjectEnvironmentMapper(IBaseMapper):
    """Mapper for Environment schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Environment) -> ProjectEnvironmentDB:
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
                for robot in db_schema.robots
            ]
        )

        camera_ids_json = json.dumps([str(camera_id) for camera_id in db_schema.camera_ids])

        return ProjectEnvironmentDB(
            id=str(db_schema.id),
            project_id="",  # Will be set by repository
            name=db_schema.name,
            robots=robots_json,
            camera_ids=camera_ids_json,
            created_at=db_schema.created_at,
            updated_at=db_schema.updated_at,
        )

    @staticmethod
    def from_schema(model: ProjectEnvironmentDB) -> Environment:
        """Convert Environment db entity to schema."""
        # Parse robots JSON
        robots_data = ProjectEnvironmentMapper._parse_json(model.robots, [])
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
        camera_ids_data = ProjectEnvironmentMapper._parse_json(model.camera_ids, [])
        camera_ids = [UUID(cid) for cid in camera_ids_data]

        return Environment(
            id=model.id,
            name=model.name,
            robots=robots,
            camera_ids=camera_ids,
            created_at=model.created_at,
            updated_at=model.updated_at,
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
