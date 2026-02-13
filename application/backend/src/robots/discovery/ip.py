import asyncio
import sys

from schemas import Robot


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
        if not robot.connection_string:
            return False
        return await self.ping(robot.connection_string)
