from db.schema import ProjectRobotDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from robots.catalog.registry import RobotCatalogRegistry
from schemas.robot import Robot, UnavailableRobot


class ProjectRobotMapper(IBaseMapper[ProjectRobotDB, Robot | UnavailableRobot]):
    """Mapper for Robot schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Robot) -> ProjectRobotDB:
        """Convert Robot schema to db model."""
        return ProjectRobotDB(
            id=str(db_schema.id),
            name=db_schema.name,
            type=db_schema.type,
            payload=db_schema.payload.model_dump(),
        )

    @staticmethod
    def from_schema(
        model: ProjectRobotDB,
        catalog_registry: RobotCatalogRegistry | None = None,
    ) -> Robot | UnavailableRobot:
        """Convert Robot db entity to schema."""
        catalog_registry = catalog_registry or RobotCatalogRegistry()
        robot = {
            "id": model.id,
            "name": model.name,
            "type": model.type,
            "payload": model.payload,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }
        if catalog_registry.get_definition(model.type) is None:
            return UnavailableRobot.model_validate(robot)
        return catalog_registry.get_robot_adapter().validate_python(robot)
