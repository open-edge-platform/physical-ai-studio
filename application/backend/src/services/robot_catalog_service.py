from exceptions import ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry
from robots.catalog.types import RobotCatalogDefinition
from schemas.robot import RobotType


class RobotCatalogService:
    def __init__(self) -> None:
        self._registry: RobotCatalogRegistry = RobotCatalogRegistry()

    def list_entries(self) -> list[RobotCatalogDefinition]:
        return self._registry.list_definitions()

    def get_definition(self, robot_type: RobotType) -> RobotCatalogDefinition:
        definition = self._registry.get_definition(robot_type)
        if definition is None:
            raise ResourceNotFoundError(
                resource_type=ResourceType.ROBOT,
                resource_id=robot_type,
                message="Robot type is not part of the catalog.",
            )
        return definition
