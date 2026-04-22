import asyncio
import sys

from schemas import Robot
from schemas.robot import SO101Robot, TrossenBimanualRobot, TrossenSingleArmRobot


class IPDiscovery:
    @staticmethod
    async def ping(ip: str, ping_timeout: float = 1.0) -> bool:
        """Async ping using system ping command.
        Works on macOS/Linux/Windows.
        """
        param = "-n" if sys.platform.lower().startswith("win") else "-c"
        command = ["ping", param, "1", "-W", str(int(ping_timeout * 1000)), ip]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return (await proc.wait()) == 0

    async def is_reachable(self, robot: Robot) -> bool:
        if not isinstance(robot, SO101Robot | TrossenSingleArmRobot):
            return False
        if not robot.payload.connection_string:
            return False
        return await self.ping(robot.payload.connection_string)

    async def is_reachable_bimanual(self, robot: Robot) -> bool:
        """Ping both arms of a bimanual robot; returns True only if both are reachable."""
        if not isinstance(robot, TrossenBimanualRobot):
            return False
        left = robot.payload.connection_string_left
        right = robot.payload.connection_string_right
        if not left or not right:
            return False
        left_ok, right_ok = await asyncio.gather(self.ping(left), self.ping(right))
        return left_ok and right_ok
