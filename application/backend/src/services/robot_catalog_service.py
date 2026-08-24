from physicalai_studio_plugin import RobotCatalogDefinition

from exceptions import ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry


class RobotCatalogService:
    def __init__(self) -> None:
        self._registry: RobotCatalogRegistry = RobotCatalogRegistry()

    @property
    def registry(self) -> RobotCatalogRegistry:
        """Return the process-wide catalog registry."""
        return self._registry

    def list_entries(self) -> list[RobotCatalogDefinition]:
        return self._registry.list_definitions()

    def get_definition(self, robot_type: str) -> RobotCatalogDefinition:
        definition = self._registry.get_definition(robot_type)
        if definition is None:
            raise ResourceNotFoundError(
                resource_type=ResourceType.ROBOT,
                resource_id=robot_type,
                message="Robot type is not part of the catalog.",
            )
        return definition
