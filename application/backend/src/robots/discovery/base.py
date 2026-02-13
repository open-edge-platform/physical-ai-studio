from abc import ABC, abstractmethod

from schemas import Robot


class RobotDiscovery(ABC):
    @abstractmethod
    async def is_reachable(self, robot: Robot) -> bool:
        pass
