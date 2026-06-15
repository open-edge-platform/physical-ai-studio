from abc import ABC, abstractmethod
from datetime import datetime

from schemas.robot import RobotType


class RobotClient(ABC):
    """Abstract interface for robot communication (commands only)."""

    name: str

    @property
    @abstractmethod
    def robot_type(self) -> RobotType:
        """Specify the RobotType"""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot is connected."""

    @abstractmethod
    def connect(self) -> None:
        """Connect to the robot."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""

    @abstractmethod
    def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""

    @abstractmethod
    def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        """Set joint positions. Returns event dict with timestamp."""

    @abstractmethod
    def enable_torque(self) -> dict:
        """Enable torque. Returns event dict with timestamp."""

    @abstractmethod
    def disable_torque(self) -> dict:
        """Disable torque. Returns event dict with timestamp."""

    @abstractmethod
    def read_state(self, *, normalize: bool = True) -> dict:
        """Read current robot state. Returns state dict with timestamp."""

    @abstractmethod
    def read_forces(self) -> dict | None:
        """Read current robot forces. Returns state dict with timestamp."""

    @abstractmethod
    def set_forces(self, forces: dict) -> dict:
        """Set current robot forces. Returns event dict with timestamp."""

    @abstractmethod
    def features(self) -> list[str]:
        """Get Robot features. Returns list with joints."""

    @staticmethod
    def _timestamp() -> float:
        """Get current timestamp in seconds since epoch."""
        return datetime.now().timestamp()

    @staticmethod
    def _create_event(event: str, **kwargs) -> dict:
        """Create an event dict with timestamp."""
        return {
            "event": event,
            "timestamp": RobotClient._timestamp(),
            **kwargs,
        }
