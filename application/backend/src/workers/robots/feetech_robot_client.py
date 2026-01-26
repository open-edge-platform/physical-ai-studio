from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from loguru import logger

from .robot_client import RobotClient


class FeetechRobotClient(RobotClient):
    """
    Implementation of RobotClient for Feetech bus-connected robots.

    Supports SO101, LeKiWi, and Reachy2 robots using the Feetech motor bus.
    """

    def __init__(
        self,
        config: SO101FollowerConfig,
        normalize: bool = True,
    ) -> None:
        """
        Initialize the Feetech robot client.

        Args:
            config: Configuration for the SO101 follower.
            normalize: Default normalization setting for joint values.
        """
        self.config = config
        self.normalize = normalize
        self.robot = SO101Follower(config)
        self.is_controlled = False

    async def connect(self) -> None:
        """
        Establish connection to the robot motor bus.

        Raises:
            Exception: If the connection to the motor bus fails.
        """
        logger.info(f"Connecting to Feetech robot on port {self.config.port}")
        try:
            self.robot.bus.connect()
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Close the connection to the robot motor bus.
        """
        logger.info(f"Disconnecting from robot on port {self.config.port}")
        if self.robot.bus.is_connected:
            self.robot.bus.disconnect()

    async def ping(self) -> dict:
        """
        Send a ping to verify communication.

        Returns:
            A 'pong' event dictionary.
        """
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict) -> dict:
        """
        Set target positions for multiple joints.

        Automatically enables torque if it's currently disabled.

        Args:
            joints: Dictionary mapping joint names to target positions.

        Returns:
            Confirmation event dictionary.
        """
        if not self.is_controlled:
            await self.enable_torque()

        self.robot.send_action({f"{name}.pos": joints[name] for name in joints})
        return self._create_event(
            "joints_state_was_set",
            joints=joints,
        )

    async def enable_torque(self) -> dict:
        """
        Enable torque on all motors.

        Returns:
            Confirmation event dictionary.
        """
        logger.info("Enabling torque")
        self.is_controlled = True
        self.robot.bus.enable_torque()
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        """
        Disable torque on all motors.

        Returns:
            Confirmation event dictionary.
        """
        logger.info("Disabling torque")
        self.is_controlled = False
        self.robot.bus.disable_torque()
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:
        """
        Read the current position of all joints.

        Args:
            normalize: Whether to return normalized values.

        Returns:
            Event dictionary containing the current joint states.
        """
        try:
            state = self.robot.bus.sync_read("Present_Position", normalize=normalize)
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise
