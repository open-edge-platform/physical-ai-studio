from serial.tools import list_ports

from schemas import Robot


class SerialDiscovery:
    async def is_reachable(self, robot: Robot) -> bool:
        ports = {p.device for p in list_ports.comports()}
        return robot.connection_string in ports
