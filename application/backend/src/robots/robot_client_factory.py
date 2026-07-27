from robots.catalog.registry import RobotCatalogRegistry
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from robots.robot_client import RobotClient
from schemas import SerialPortInfo
from schemas.robot import Robot
from utils.serial_robot_tools import RobotConnectionManager


class RobotClientFactory:
    robot_manager: RobotConnectionManager
    catalog_registry: RobotCatalogRegistry

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        catalog_registry: RobotCatalogRegistry | None = None,
    ) -> None:
        self.robot_manager = robot_manager
        self.catalog_registry = catalog_registry or RobotCatalogRegistry()

    async def build(self, robot: Robot) -> RobotClient:
        definition = self.catalog_registry.get_definition(robot.type)

        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot.type}")

        builder = definition.robot_builder
        if builder is None:
            raise ValueError(f"Robot type {robot.type} has no robot builder")

        robot_driver = await builder(robot, self)
        adapter_options = definition.adapter_options
        return PhysicalAIRobotAdapter(
            robot=robot_driver,
            robot_type=robot.type,
            robot_role=definition.role,
            config=PhysicalAIRobotAdapterConfig(
                include_velocities=adapter_options.include_velocities,
                goal_time_scale=adapter_options.goal_time_scale,
                external_effort_gain=adapter_options.external_effort_gain,
            ),
        )

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        port = self._resolve_port(self.robot_manager.robots, port_info)
        if port is not None:
            return port

        await self.robot_manager.find_robots()
        return self._resolve_port(self.robot_manager.robots, port_info)

    @staticmethod
    def _resolve_port(discovered: list[SerialPortInfo], target: SerialPortInfo) -> str | None:
        if target.serial_number:
            for serial_port in discovered:
                if serial_port.serial_number == target.serial_number:
                    return serial_port.connection_string
            return None

        for serial_port in discovered:
            if serial_port.connection_string == target.connection_string:
                return serial_port.connection_string
        return None
