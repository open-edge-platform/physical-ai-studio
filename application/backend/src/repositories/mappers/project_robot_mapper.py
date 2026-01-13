from db.schema import ProjectRobotDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.robot import Robot 


class ProjectRobotMapper(IBaseMapper):
    """Mapper for Robot schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Robot) -> ProjectRobotDB:
        """Convert Robot schema to db model."""
        return ProjectRobotDB(
            id=str(db_schema.id),
            name=db_schema.name,
            serial_id=db_schema.serial_id,
            type=db_schema.type,
        )

    @staticmethod
    def from_schema(model: ProjectRobotDB) -> Robot:
        """Convert Robot db entity to schema."""
        return Robot(
            id=model.id,
            name=model.name,
            serial_id=model.serial_id,
            type=model.type,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
