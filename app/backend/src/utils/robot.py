import lerobot
import time
from typing import Any, List, Set, Optional

from schemas import RobotPortInfo
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
available_ports = list_ports.comports()


def from_port(port: ListPortInfo, device_name: str) -> Optional[RobotPortInfo]:
    """
    Detect if the device is a SO-100 robot.π
    """
    # The Feetech UART board CH340 has PID 29987
    if port.pid == 21971 or port.pid == 29987:
            # The serial number is not always available
            serial_number = port.serial_number or "no_serial"
            robot = RobotPortInfo(
            port=port.device,
            serial_id=serial_number,
            device_name= device_name
            )
            return robot
    return None

class RobotConnectionManager:
    _all_robots: List[RobotPortInfo] = []
    available_ports: List[ListPortInfo]

    def __init__(self):
        self.available_ports = list_ports.comports()

    @property
    def robots(self) -> List[RobotPortInfo]:
        return self._all_robots

    async def find_robots(self) -> None:
        """
        Loop through all available ports and try to connect to a robot.

        Use self.scan_ports() before to update self.available_ports and self.available_can_ports
        """

        self._all_robots = []

        # If we are only simulating, we can just use the SO100Hardware class
        # Keep track of connected devices by port name and serial to avoid duplicates
        connected_devices: Set[str] = set()
        connected_serials: Set[str] = set()

        # Try each serial port exactly once
        for port in self.available_ports:
            serial_num = getattr(port, "serial_number", None)
            # Skip if this port or its serial has already been connected
            if port.device in connected_devices or (
                serial_num and serial_num in connected_serials
            ):
                print(f"Skipping {port.device}: already connected (or alias).")
                continue

            for name in [
                "so-100",
            ]:
                print(
                    f"Trying to connect to {name} on {port.device}."
                )
                robot = from_port(port, device_name=name)
                if robot is None:
                    print(
                        f"Failed to create robot from {name} on {port.device}."
                    )
                    continue
                print(f"Robot created: {robot}")
                #await robot.connect()

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

async def find_robots() -> List[RobotPortInfo]:
    manager = RobotConnectionManager()
    await manager.find_robots()
    return manager.robots


async def identify_robot_visually(robot: RobotPortInfo, joint: str | None = None):
    if robot.device_name != "so-100":
        raise ValueError(f"Trying to identify unsupported robot: {robot.device_name}")

    if joint is None:
        joint = "gripper"

    #Assume follower since leader shares same FeetechMotorBus layout
    robot = SO101Follower(SO101FollowerConfig(port=robot.port))
    robot.bus.connect()

    PRESENT_POSITION_KEY = "Present_Position"
    GOAL_POSITION_KEY = "Goal_Position"

    current_position = robot.bus.sync_read(PRESENT_POSITION_KEY, normalize=False)
    gripper_calibration = robot.bus.read_calibration()[joint]
    robot.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_min, normalize=False)
    time.sleep(1)
    robot.bus.write(GOAL_POSITION_KEY, joint, gripper_calibration.range_max, normalize=False)
    time.sleep(1)
    robot.bus.write(GOAL_POSITION_KEY, joint, current_position[joint], normalize=False)
    time.sleep(1)
    robot.bus.disconnect()
