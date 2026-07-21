from uuid import UUID

from exceptions import ResourceNotFoundError, ResourceType
from robots.catalog.registry import RobotCatalogRegistry
from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from robots.robot_client import RobotClient
from schemas.calibration import Calibration
from schemas.robot import Robot, SO101Robot
from services.robot_calibration_service import RobotCalibrationService
from utils.serial_robot_tools import RobotConnectionManager, find_so101_port, serial_port_from_so101


class RobotClientFactory:
    calibration_service: RobotCalibrationService
    robot_manager: RobotConnectionManager
    catalog_registry: RobotCatalogRegistry

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
        catalog_registry: RobotCatalogRegistry | None = None,
    ) -> None:
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service
        self.catalog_registry = catalog_registry or RobotCatalogRegistry()

    async def build(self, robot: Robot) -> RobotClient:
        definition = self.catalog_registry.get_definition(robot.type)

        if definition is None:
            raise ValueError(f"Robot type is not part of the catalog: {robot.type}")

        builder = definition.robot_builder

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

    async def find_so101_port(self, robot: SO101Robot) -> str:
        port = await find_so101_port(self.robot_manager, serial_port_from_so101(robot))
        if port is None:
            resource_key = robot.payload.serial_number or robot.payload.connection_string
            raise ResourceNotFoundError(ResourceType.ROBOT, resource_key)
        return port

    # NOTE: this is a bit awkward due to how SO101 is implemented,
    # perhaps we can add the Calibration into the SO101RobotPayload
    async def get_robot_calibration(self, robot: SO101Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None
        return await self.calibration_service.get_calibration(robot.active_calibration_id)

    async def find_port_by_serial(self, serial_number: str) -> str | None:
        for managed_robot in self.robot_manager.robots:
            if managed_robot.serial_number == serial_number:
                return managed_robot.connection_string
        return None

    async def get_calibration_by_id(self, calibration_id: UUID | None) -> Calibration | None:
        if calibration_id is None:
            return None
        return await self.calibration_service.get_calibration(calibration_id)
