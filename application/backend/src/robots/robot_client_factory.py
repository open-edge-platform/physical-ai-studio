from typing import Any

from loguru import logger
from physicalai.robot import SharedRobot
from physicalai_studio_plugin import CatalogRobotFactory, RobotCatalogDefinition, shared_robot_name

from exceptions import RobotPluginUnavailableError
from robots.catalog.registry import RobotCatalogRegistry
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from robots.robot_client import RobotClient
from schemas import SerialPortInfo
from schemas.robot import ReadableRobot, UnavailableRobot
from utils.device_paths import resolve_serial_device
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

    async def build(self, robot: ReadableRobot) -> RobotClient:
        if isinstance(robot, UnavailableRobot):
            raise RobotPluginUnavailableError(robot.name, robot.type)

        shared_robot, definition = await self.build_shared_robot(robot)
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

    async def build_robot_driver(
        self, robot: ReadableRobot, port_finder: CatalogRobotFactory
    ) -> tuple[Any, RobotCatalogDefinition]:
        """Run the catalog builder for a robot and return its driver and definition."""
        if isinstance(robot, UnavailableRobot):
            raise RobotPluginUnavailableError(robot.name, robot.type)

        definition = self.catalog_registry.get_definition(robot.type)
        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot.type}")
        if definition.robot_builder is None:
            raise ValueError(f"Robot type {robot.type} has no robot builder")

        return await definition.robot_builder(robot, port_finder), definition

    async def build_shared_robot(self, robot: ReadableRobot) -> tuple[SharedRobot, RobotCatalogDefinition]:
        """Build the shared transport used by runtime and adapter callers."""
        robot_driver, definition = await self.build_robot_driver(robot, self)
        # Builders return a plain driver; wrapping happens here so every robot
        # type (including third-party plugins) gets one owner process holding
        # the hardware. The driver itself is discarded — only its recipe is sent,
        # and the owner rebuilds it. The name keys the owner's Zenoh topics, so
        # it must come from the id, never the free-form display name.
        shared_robot = SharedRobot.from_config(robot_driver, name=shared_robot_name(robot.id))
        return shared_robot, definition

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        """Resolve a live serial port for a robot whose identity is known.

        Returns ``None`` when the device is not attached, which the catalog
        builders turn into an error. Reusing the port stored at registration
        time would connect to whatever device now owns that path; the one
        caller that deliberately wants the stored value is the runtime config
        export (see ``runtime.config_builder``).

        A ``/dev/serial/by-id`` alias is only substituted when the robot was
        matched by serial number. Without one the match is by path alone, so
        the device was located rather than identified, and a stable-looking
        alias would dress that guess up as a verified one.
        """
        port = await self.robot_manager.find_port(port_info)
        if port is None:
            return None
        if not port_info.serial_number:
            logger.warning(
                "Robot at {} has no serial number stored; matched by path, so its identity is unverified.",
                port,
            )
            return port
        return resolve_serial_device(port)
