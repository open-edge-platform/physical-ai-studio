import asyncio

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from loguru import logger
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from schemas import Robot, SerialPortInfo
from schemas.robot import RobotType

available_ports = list_ports.comports()


def from_port(port: ListPortInfo, robot_type: str) -> SerialPortInfo | None:
    """Detect if the device is a SO-100 robot."""
    # The Feetech UART board CH340 has PID 29987
    if port.pid in {21971, 29987}:
        # The serial number is not always available
        serial_number = port.serial_number or "no_serial"
        return SerialPortInfo(connection_string=port.device, serial_number=serial_number, robot_type=robot_type)
    return None


class RobotConnectionManager:
    _all_robots: list[SerialPortInfo] = []
    available_ports: list[ListPortInfo]

    def __init__(self):
        self.available_ports = list_ports.comports()

    @property
    def robots(self) -> list[SerialPortInfo]:
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
                logger.debug(f"Skipping {port.device}: already connected (or alias).")
                continue

            for name in [
                "so-100",
            ]:
                # logger.debug(f"Trying to connect to {name} on {port.device}.")
                robot = from_port(port, robot_type=name)
                if robot is None:
                    # logger.debug(f"Failed to create robot from {name} on {port.device}.")
                    continue
                logger.debug(f"Robot created: {robot}")
                # await robot.connect()

                if robot is not None:
                    logger.debug(f"Connected to {name} on {port.device}.")
                    self._all_robots.append(robot)
                    # Mark both device and serial as connected
                    connected_devices.add(port.device)
                    if serial_num:
                        connected_serials.add(serial_num)
                    break  # stop trying other classes on this port

        if not self._all_robots:
            logger.debug("No robot connected.")


async def find_robots() -> list[SerialPortInfo]:
    """Find all robots connected via serial"""
    manager = RobotConnectionManager()
    await manager.find_robots()
    return manager.robots


def find_port_for_serial(serial_number: str) -> str:
    """Find the serial port path given the serial number of the device"""
    ports = list_ports.comports()
    for port in ports:
        serial_num = getattr(port, "serial_number", None)
        if serial_num == serial_number:
            return port.device
    return ""


async def identify_so101_robot_visually(robot: Robot, joint: str | None = None) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position"""
    if robot.type not in {RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    if joint is None:
        joint = "gripper"

    if robot.connection_string == "" and robot.serial_number != "":
        robot.connection_string = find_port_for_serial(robot.serial_number)

    if robot.connection_string == "":
        raise ValueError(f"Could not find the serial port for serial number {robot.serial_number}")
    # Assume follower since leader shares same FeetechMotorBus layout
    connection = SO101Follower(SO101FollowerConfig(port=robot.connection_string))
    connection.bus.connect()

    PRESENT_POSITION_KEY = "Present_Position"
    GOAL_POSITION_KEY = "Goal_Position"

    current_position = connection.bus.sync_read(PRESENT_POSITION_KEY, normalize=False)
    gripper_calibration = connection.bus.read_calibration()[joint]
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_min, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_max, normalize=False)
    await asyncio.sleep(1)
    connection.bus.write(GOAL_POSITION_KEY, joint, current_position[joint], normalize=False)
    await asyncio.sleep(1)
    connection.bus.disconnect()
