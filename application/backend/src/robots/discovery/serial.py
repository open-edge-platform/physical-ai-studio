from serial.tools import list_ports

from schemas import Robot
from schemas.robot import SO101Robot, TrossenSingleArmRobot


class SerialDiscovery:
    async def is_reachable(self, robot: Robot) -> bool:
        if not isinstance(robot, SO101Robot | TrossenSingleArmRobot):
            return False
        ports = {p.device for p in list_ports.comports()}
        return robot.payload.connection_string in ports
