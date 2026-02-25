import asyncio

from lerobot.robots.so101_follower import SO101Follower as LeSO101Follower
from lerobot.robots.so101_follower import SO101FollowerConfig
from loguru import logger

from robots.robot_client import RobotClient
from schemas.robot import RobotType

# Timeout for hardware operations (seconds)
# Connection may take longer due to USB enumeration
HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0


class SO101Follower(RobotClient):
    robot: LeSO101Follower
    name = "so101_follower"

    max_speed = 270  # From feetech 12V servo spec: 60 deg / 0.222s

    previous_target: dict[str, float] | None = None

    def __init__(self, config: SO101FollowerConfig):
        self.robot = LeSO101Follower(config)
        self.is_controlled = False
        # Serialize all serial bus access. The Feetech SCS bus is half-duplex
        # (single TX/RX line), so concurrent reads/writes cause "Port is in use!"
        # errors. This lock ensures only one bus operation is in-flight at a time.
        self._bus_lock = asyncio.Lock()

    @property
    def robot_type(self) -> RobotType:
        return RobotType.SO101_FOLLOWER

    @property
    async def is_connected(self) -> bool:
        return self.robot.is_connected

    async def connect(self) -> None:
        """Connect to the robot."""
        logger.info(f"Connecting to SO101Follower on port {self.robot.config.port}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_CONNECT):
                await asyncio.to_thread(self.robot.connect)
        except TimeoutError:
            logger.error("Timeout connecting to robot")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the robot."""
        logger.info(f"Disconnecting SO101Follower on port {self.robot.config.port}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self.robot.disconnect)
            logger.info("Robot disconnected")
        except TimeoutError:
            logger.warning("Timeout during robot disconnect - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")

    async def ping(self) -> dict:
        """Send ping command. Returns event dict with timestamp."""
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        """Set joint positions. Returns event dict with timestamp.

        The challenge here is when the maximum degree/s * goal_time results in sub servo resulution distances.
        This means that the servo will either not move, or very slowly.
        In order to fix this we store the previous target so that previous attempts to move it to a sub servo
        resolution position so that these small steps can accumulate.

        However, the previous_target must remain relevant and close to the current state.
        """
        max_frame_speed = self.max_speed * goal_time

        async with self._bus_lock:
            async with asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                state = await asyncio.to_thread(self.robot.get_observation)

            if self.previous_target:
                # Additional clamp to make sure that previous_target is not too far of current position
                state = self._clamp_joints(state, self.previous_target, max_frame_speed * 2)

            target = {
                key: value + self._clamp_speed(joints[key] - value, max_frame_speed) for key, value in state.items()
            }
            self.previous_target = target

            if not self.is_controlled:
                await self._enable_torque_unlocked()

            async with asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self.robot.send_action, target)

        return self._create_event(
            "joints_state_was_set",
            joints=target,
        )

    @staticmethod
    def _clamp_joints(current: dict, target: dict, max_distance: float) -> dict:
        """Clamp a current joints dict to target with a max value"""
        return {
            key: value + SO101Follower._clamp_speed(target[key] - value, max_distance) for key, value in current.items()
        }

    @staticmethod
    def _clamp_speed(value: float, speed: float) -> float:
        """Clamp value between -speed and speed."""
        return max(min(value, speed), -speed)

    async def enable_torque(self) -> dict:
        """Enable torque. Returns event dict with timestamp."""
        logger.info("Enabling torque")
        async with self._bus_lock:
            await self._enable_torque_unlocked()
        return self._create_event("torque_was_enabled")

    async def _enable_torque_unlocked(self) -> None:
        """Enable torque without acquiring the bus lock.

        Must be called while holding self._bus_lock.
        """
        async with asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self.robot.bus.enable_torque)
        self.is_controlled = True

    async def disable_torque(self) -> dict:
        """Disable torque. Returns event dict with timestamp."""
        logger.info("Disabling torque")
        async with self._bus_lock:
            await self._disable_torque_unlocked()
        return self._create_event("torque_was_disabled")

    async def _disable_torque_unlocked(self) -> None:
        """Disable torque without acquiring the bus lock.

        Must be called while holding self._bus_lock.
        """
        async with asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self.robot.bus.disable_torque)
        self.is_controlled = False

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
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                state = await asyncio.to_thread(self.robot.get_observation)
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
