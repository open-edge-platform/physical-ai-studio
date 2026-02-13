from robots.discovery.ip import IPDiscovery
from robots.discovery.serial import SerialDiscovery
from schemas import Robot
from schemas.robot import RobotType
from utils.serial_robot_tools import RobotConnectionManager


class DiscoveryManager:
    def __init__(self):
        self.serial = SerialDiscovery()
        self.ip = IPDiscovery()
        self.serial_manager = RobotConnectionManager()

    async def refresh_hardware_ports(self) -> None:
        await self.serial_manager.find_robots()

    async def is_robot_online(self, robot: Robot) -> bool:
        if robot.type in {RobotType.SO101_LEADER, RobotType.SO101_FOLLOWER}:
            return robot.serial_number in [cs.serial_number for cs in self.serial_manager.robots]
        if robot.type in {RobotType.TROSSEN_WIDOWXAI_LEADER, RobotType.TROSSEN_WIDOWXAI_FOLLOWER}:
            return await self.ip.is_reachable(robot)
        return False
