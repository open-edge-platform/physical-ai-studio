from lerobot.robots.so101_follower import SO101Follower as LeSO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
from loguru import logger

from robots.robot_client import RobotClient
from schemas.robot import RobotType


class SO101Follower(RobotClient):
    robot: LeSO101Follower
    name = "so101_follower"

    def __init__(self, config: SO101FollowerConfig):
        self.robot = LeSO101Follower(config)
        self.is_controlled = False

    @property
    def robot_type(self) -> RobotType:
        return RobotType.SO101_FOLLOWER

    @property
    async def is_connected(self) -> bool:
        return self.robot.is_connected

    async def connect(self) -> None:
        """Connect to the robot."""
        logger.info(f"Connecting to SO101Follower on port {self.robot.config.port}")
        self.robot.connect()

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        logger.info(f"Disconnecting to SO101Follower on port {self.robot.config.port}")
        self.robot.disconnect()

    async def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        """Set joint positions. Returns event dict with timestamp."""
        self.robot.send_action(joints)
        return self._create_event(
            "joints_state_was_set",
            joints=joints,
        )

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
        """Read current robot state. Returns state dict with timestamp.

        Example state: {
            'shoulder_pan.pos': -8.705526116578355,
            'shoulder_lift.pos': -98.16753926701571,
            'elbow_flex.pos': 95.98393574297188,
            'wrist_flex.pos': 73.85993485342019,
            'wrist_roll.pos': -13.84615384615384,
            'gripper.pos': 26.885644768856448
        }
        """
        try:
            state = self.robot.get_observation()
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
