from physicalai.config import to_config
from physicalai.robot import SharedRobot
from physicalai_studio_plugin import shared_robot_name

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
        # Builders return a plain driver; wrapping happens here so every robot
        # type (including third-party plugins) gets one owner process holding
        # the hardware. The driver itself is discarded — only its recipe is sent,
        # and the owner rebuilds it. The name keys the owner's Zenoh topics, so
        # it must come from the id, never the free-form display name.
        shared_robot = SharedRobot.from_config(to_config(robot_driver), name=shared_robot_name(robot.id))
        adapter_options = definition.adapter_options
        return PhysicalAIRobotAdapter(
            robot=shared_robot,
            robot_type=robot.type,
            robot_role=definition.role,
            display_name=robot.name,
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
