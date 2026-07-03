import asyncio

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from loguru import logger
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from schemas import Robot, SerialPortInfo
from schemas.robot import SO101Robot


def serial_port_from_so101(robot: SO101Robot) -> SerialPortInfo:
    """Build a serial identity from an SO101 robot configuration."""
    connection_string = robot.payload.connection_string or None
    serial_number = robot.payload.serial_number or None
    return SerialPortInfo(connection_string=connection_string, serial_number=serial_number)


def from_port(port: ListPortInfo) -> SerialPortInfo | None:
    """Detect if the device is a SO-100 robot."""
    serial_number = getattr(port, "serial_number", None)

    # Ignore internal hardware
    if port.device.startswith("/dev/ttyS"):
        return None

    # The Feetech UART board CH340 has PID 29987
    if port.pid not in {21971, 29987}:
        logger.debug("Found usb port with unexpected PID, {device}: {pid}", device=port.device, pid=port.pid)
        return None

    return SerialPortInfo(connection_string=port.device, serial_number=serial_number or None)


class RobotConnectionManager:
    _all_robots: list[SerialPortInfo]
    available_ports: list[ListPortInfo]

    def __init__(self):
        self.available_ports = list(list_ports.comports())
        self._all_robots = []

    @property
    def robots(self) -> list[SerialPortInfo]:
        return self._all_robots

    async def find_robots(self) -> None:
        """
        Loop through all available ports and try to connect to a robot.

        Use self.scan_ports() before to update self.available_ports and self.available_can_ports
        """
        self.available_ports = list(list_ports.comports())
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

            robot = from_port(port)
            if robot is None:
                continue

            logger.debug(f"Robot created: {robot}")
            self._all_robots.append(robot)
            connected_devices.add(port.device)
            if serial_num:
                connected_serials.add(serial_num)

        if not self._all_robots:
            logger.debug("No robot connected.")


def _resolve_serial_port(discovered: list[SerialPortInfo], target: SerialPortInfo) -> str | None:
    if target.serial_number is not None:
        for serial_port in discovered:
            if serial_port.serial_number == target.serial_number:
                return serial_port.connection_string
        return None

    for serial_port in discovered:
        if serial_port.connection_string == target.connection_string:
            return serial_port.connection_string
    return None


async def find_so101_port(
    manager: RobotConnectionManager,
    serial_port: SerialPortInfo,
) -> str | None:
    """Find the current port for an SO101 robot by serial number or configured port."""
    port = _resolve_serial_port(manager.robots, serial_port)
    if port is not None:
        return port

    await manager.find_robots()
    return _resolve_serial_port(manager.robots, serial_port)


async def identify_so101_robot_visually(
    manager: RobotConnectionManager,
    robot: Robot,
    joint: str | None = None,
) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position"""
    if not isinstance(robot, SO101Robot):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    if joint is None:
        joint = "gripper"

    connection_string = await find_so101_port(manager, serial_port_from_so101(robot))

    if connection_string is None:
        if robot.payload.serial_number:
            raise ValueError(f"Could not find the serial port for serial number {robot.payload.serial_number}")
        raise ValueError("Could not resolve a serial port from connection_string")
    # Assume follower since leader shares same FeetechMotorBus layout
    connection = SOFollower(SOFollowerRobotConfig(port=connection_string))
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
