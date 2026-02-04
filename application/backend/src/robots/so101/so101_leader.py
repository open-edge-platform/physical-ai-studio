from lerobot.teleoperators.so101_leader import SO101Leader as LeSO101Leader
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from loguru import logger

from robots.robot_client import RobotClient


class SO101Leader(RobotClient):
    robot: LeSO101Leader
    name = "so101_leader"
    is_controlled: bool = False

    def __init__(self, port: str, id: str):
        config = SO101LeaderConfig(port=port, id=id)
        self.robot = LeSO101Leader(config)

    async def is_connected(self) -> bool:
        return self.robot.is_connected

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

    async def set_joints_state(self, joints: dict) -> dict:  # noqa: ARG002
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

    def features(self) -> list[str]:
        """Get Robot features. Returns list with joints."""
        return list(self.robot.action_features.keys())

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        """Read current robot state. Returns state dict with timestamp."""
        try:
            observation = self.robot.get_action()
            state = {key.removesuffix(".pos"): value for key, value in observation.items()}
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        """Read current robot forces. Returns state dict with timestamp."""
        return self._create_event(
            "force_was_updated",
            state=None,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:  # noqa: ARG002
        """Set current robot forces. Returns event dict with timestamp."""
        raise Exception("Not implemented for SO101")
