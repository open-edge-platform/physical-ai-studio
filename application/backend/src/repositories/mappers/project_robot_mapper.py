import json

from db.schema import ProjectRobotDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.robot import Robot, RobotCamera


class ProjectRobotMapper(IBaseMapper):
    """Mapper for Robot schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Robot) -> ProjectRobotDB:
        """Convert Robot schema to db model."""
        cameras_json = json.dumps([camera.model_dump() for camera in db_schema.cameras])

        return ProjectRobotDB(
            id=str(db_schema.id),
            name=db_schema.name,
            serial_id=db_schema.serial_id,
            type=db_schema.type,
            cameras=cameras_json,
        )

    @staticmethod
    def from_schema(model: ProjectRobotDB) -> Robot:
        """Convert Robot db entity to schema."""
        cameras_data = []
        try:
            if isinstance(model.cameras, str):
                cameras_data = json.loads(model.cameras)
            else:
                cameras_data = model.cameras
        except (json.JSONDecodeError, TypeError):
            cameras_data = []

        cameras = [RobotCamera(**camera) for camera in cameras_data]

        return Robot(
            id=model.id,
            name=model.name,
            serial_id=model.serial_id,
            type=model.type,
            cameras=cameras,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
