from lerobot.teleoperators.so101_leader import SO101Leader as LeSO101Leader, SO101LeaderConfig
from schemas import Robot
from datetime import datetime
from loguru import logger

from workers.robots.robot_client import RobotClient


class SO101Leader(RobotClient):
    robot: LeSO101Leader
    is_controlled: bool = False

    def __init__(self, config: SO101LeaderConfig):
        self.robot = LeSO101Leader(config)

    async def connect(self) -> None:
        """Connect to the robot."""
        logger.info(f"Connecting to SO101Leader on port {self.robot.config.port}")
        self.robot.connect()

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        logger.info(f"Disconnecting to SO101Leader on port {self.robot.config.port}")
        self.robot.disconnect()

    async def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        """Set joint positions. Returns event dict with timestamp."""
        raise Exception("Not implemented for leaders")

    async def enable_torque(self) -> dict:
        """Enable torque. Returns event dict with timestamp."""
        logger.info("Enabling torque")
        self.is_controlled = True
        self.robot.bus.enable_torque()
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """Disable torque. Returns event dict with timestamp."""
        logger.info("Disabling torque")
        self.is_controlled = False
        self.robot.bus.disable_torque()
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:
        """Read current robot state. Returns state dict with timestamp."""
        try:
            state = self.robot.get_action()
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

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
