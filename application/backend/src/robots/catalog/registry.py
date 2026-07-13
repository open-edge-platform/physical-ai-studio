from schemas.robot import RobotType

from . import so101, widowxai
from .types import RobotCatalogDefinition


class RobotCatalogRegistry:
    def __init__(self) -> None:
        self._definitions: dict[RobotType, RobotCatalogDefinition] = {}

        for definition in so101.get_definitions() + widowxai.get_definitions():
            self.register(definition)

    def list_definitions(self) -> list[RobotCatalogDefinition]:
        return list(self._definitions.values())

    def get_definition(self, robot_type: RobotType) -> RobotCatalogDefinition | None:
        return self._definitions.get(robot_type)

    def register(self, definition: RobotCatalogDefinition) -> None:
        if definition.robot_type in self._definitions:
            raise ValueError(f"Duplicate robot catalog registration for type: {definition.robot_type}")

        self._definitions[definition.robot_type] = definition
