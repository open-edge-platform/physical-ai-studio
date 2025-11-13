import asyncio

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig as LeRobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig as LeRobotTeleoperatorConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from schemas import RobotConfig, RobotPortInfo

available_ports = list_ports.comports()


def from_port(port: ListPortInfo, robot_type: str) -> RobotPortInfo | None:
    """Detect if the device is a SO-100 robot.π"""
    # The Feetech UART board CH340 has PID 29987
    if port.pid in {21971, 29987}:
        # The serial number is not always available
        serial_number = port.serial_number or "no_serial"
        return RobotPortInfo(port=port.device, serial_id=serial_number, robot_type=robot_type)
    return None


class RobotConnectionManager:
    _all_robots: list[RobotPortInfo] = []
    available_ports: list[ListPortInfo]

    def __init__(self):
        self.available_ports = list_ports.comports()

    @property
    def robots(self) -> list[RobotPortInfo]:
        return self._all_robots

    async def find_robots(self) -> None:
        """
        Loop through all available ports and try to connect to a robot.

        Use self.scan_ports() before to update self.available_ports and self.available_can_ports
        """

        self._all_robots = []

        # If we are only simulating, we can just use the SO100Hardware class
        # Keep track of connected devices by port name and serial to avoid duplicates
        connected_devices: set[str] = set()
        connected_serials: set[str] = set()

        # Try each serial port exactly once
        for port in self.available_ports:
            serial_num = getattr(port, "serial_number", None)
            # Skip if this port or its serial has already been connected
            if port.device in connected_devices or (serial_num and serial_num in connected_serials):
                print(f"Skipping {port.device}: already connected (or alias).")
                continue

            for name in [
                "so-100",
            ]:
                print(f"Trying to connect to {name} on {port.device}.")
                robot = from_port(port, robot_type=name)
                if robot is None:
                    print(f"Failed to create robot from {name} on {port.device}.")
                    continue
                print(f"Robot created: {robot}")
                # await robot.connect()

                if robot is not None:
                    print(f"Connected to {name} on {port.device}.")
                    self._all_robots.append(robot)
                    # Mark both device and serial as connected
                    connected_devices.add(port.device)
                    if serial_num:
                        connected_serials.add(serial_num)
                    break  # stop trying other classes on this port

        if not self._all_robots:
            print("No robot connected.")


async def find_robots() -> list[RobotPortInfo]:
    """Find all robots connected via serial"""
    manager = RobotConnectionManager()
    await manager.find_robots()
    return manager.robots


async def identify_robot_visually(robot: RobotPortInfo, joint: str | None = None) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position"""
    if robot.robot_type != "so-100":
        raise ValueError(f"Trying to identify unsupported robot: {robot.robot_type}")

    if joint is None:
        joint = "gripper"

    # Assume follower since leader shares same FeetechMotorBus layout
    robot = SO101Follower(SO101FollowerConfig(port=robot.port))
    robot.bus.connect()

    PRESENT_POSITION_KEY = "Present_Position"
    GOAL_POSITION_KEY = "Goal_Position"

    current_position = robot.bus.sync_read(PRESENT_POSITION_KEY, normalize=False)
    gripper_calibration = robot.bus.read_calibration()[joint]
    robot.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_min, normalize=False)
    await asyncio.sleep(1)
    robot.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_max, normalize=False)
    await asyncio.sleep(1)
    robot.bus.write(GOAL_POSITION_KEY, joint, current_position[joint], normalize=False)
    await asyncio.sleep(1)
    robot.bus.disconnect()


def make_lerobot_robot_config_from_robot(config: RobotConfig, cameras: dict[str, CameraConfig]) -> LeRobotConfig:
    """Build LeRobot Follower Config from our RobotConfig."""
    le_config = {
        "id": config.id,
        "port": config.port,
        "cameras": cameras,
    }

    if config.robot_type == "so100_follower":
        from lerobot.robots.so100_follower import SO100FollowerConfig

        return SO100FollowerConfig(**le_config)
    if config.robot_type == "so101_follower":
        from lerobot.robots.so101_follower import SO101FollowerConfig

        return SO101FollowerConfig(**le_config)
    raise ValueError(config.type)


def make_lerobot_teleoperator_config_from_robot(config: RobotConfig) -> LeRobotTeleoperatorConfig:
    """Build LeRobot Teleoperator Config from our RobotConfig."""
    le_config = {
        "id": config.id,
        "port": config.port,
        "calibration_dir": None,
    }
    if config.robot_type == "so100_follower":
        from lerobot.teleoperators.so100_leader import SO100LeaderConfig

        return SO100LeaderConfig(**le_config)
    if config.robot_type == "so101_follower":
        from lerobot.teleoperators.so101_leader import SO101LeaderConfig

        return SO101LeaderConfig(**le_config)
    raise ValueError(config.type)
