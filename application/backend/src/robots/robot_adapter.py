"""SO-101 protocol adapter.

Wraps physicalai's ``SO101`` driver behind the backend's ``RobotClient`` ABC,
converting between physicalai's radian-based interface and the application's
normalized joint coordinates (body: [-100, 100], gripper: [0, 100]).

This adapter is SO101-specific because the calibration conversion and
normalization logic depend on SO101 joint structure.  A generic adapter can
be extracted when a second robot type migrates to physicalai.
"""

import asyncio
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.so101 import SO101
from physicalai.robot.so101.constants import MAX_SPEED_RAD_S, RADIANS_PER_TICK, SO101_JOINT_ORDER

from robots.robot_client import RobotClient
from schemas.calibration import Calibration
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0

RobotMode = Literal["follower", "teleoperator"]


def _clamp(value: float, limit: float) -> float:
    return max(min(value, limit), -limit)


def _clamp_joints(current: dict[str, float], target: dict[str, float], max_distance: float) -> dict[str, float]:
    return {key: value + _clamp(target[key] - value, max_distance) for key, value in current.items()}


class SO101Adapter(RobotClient):
    """Adapt physicalai's :class:`SO101` to the backend's :class:`RobotClient` interface.

    Unit flow:
        read:  physicalai (radians) → ticks → normalized
        write: normalized → ticks → physicalai (radians)
    """

    name = "So101"

    def __init__(
        self,
        robot: SO101,
        mode: RobotMode,
        calibration: Calibration,
    ) -> None:
        self._robot = robot
        self._mode = mode
        self._calibration = calibration
        self._bus_lock = asyncio.Lock()

        self.previous_target: dict[str, float] | None = None
        self.is_controlled: bool = False

        self._joint_params: dict[str, dict] = {}
        for name in SO101_JOINT_ORDER:
            cal_val = self._calibration.values[name]
            self._joint_params[name] = {
                "drive_mode": cal_val.drive_mode,
                "homing_offset": cal_val.homing_offset,
                "range_min": cal_val.range_min,
                "range_max": cal_val.range_max,
            }

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.SO101_FOLLOWER
        return RobotType.SO101_LEADER

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    # Normalization formulas match lerobot motors_bus.py _normalize/_unnormalize:
    #   RANGE_M100_100 (body): norm = (((ticks - min) / (max - min)) * 200) - 100
    #   RANGE_0_100 (gripper): norm = ((ticks - min) / (max - min)) * 100
    #   If drive_mode == 1, flip sign (M100) or flip to 100-norm (0_100).

    def _radians_to_ticks(self, radians: np.ndarray) -> np.ndarray:
        cal = self._robot._calibration
        if cal is None:
            msg = "Robot calibration is required for unit conversion"
            raise RuntimeError(msg)

        ticks = np.empty(len(SO101_JOINT_ORDER), dtype=np.int32)
        for i, name in enumerate(SO101_JOINT_ORDER):
            jcal = cal.joints[name]
            raw = round(radians[i] / (jcal.direction * RADIANS_PER_TICK) + jcal.homing_offset)
            ticks[i] = int(np.clip(raw, jcal.range_min, jcal.range_max))
        return ticks

    def _ticks_to_radians(self, ticks: np.ndarray) -> np.ndarray:
        cal = self._robot._calibration
        if cal is None:
            msg = "Robot calibration is required for unit conversion"
            raise RuntimeError(msg)

        result = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
        for i, name in enumerate(SO101_JOINT_ORDER):
            jcal = cal.joints[name]
            result[i] = (ticks[i] - jcal.homing_offset) * jcal.direction * RADIANS_PER_TICK
        return result

    def _ticks_to_normalized(self, ticks: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for i, name in enumerate(SO101_JOINT_ORDER):
            params = self._joint_params[name]
            rng_min = params["range_min"]
            rng_max = params["range_max"]
            drive_mode = params["drive_mode"]
            tick = float(ticks[i])

            if name == "gripper":
                # RANGE_0_100
                norm = ((tick - rng_min) / (rng_max - rng_min)) * 100.0 if rng_max != rng_min else 0.0
                if drive_mode == 1:
                    norm = 100.0 - norm
            else:
                # RANGE_M100_100
                norm = (((tick - rng_min) / (rng_max - rng_min)) * 200.0) - 100.0 if rng_max != rng_min else 0.0
                if drive_mode == 1:
                    norm = -norm

            result[f"{name}.pos"] = norm
        return result

    def _normalized_to_ticks(self, joints: dict[str, float]) -> np.ndarray:
        ticks = np.empty(len(SO101_JOINT_ORDER), dtype=np.int32)
        for i, name in enumerate(SO101_JOINT_ORDER):
            params = self._joint_params[name]
            rng_min = params["range_min"]
            rng_max = params["range_max"]
            drive_mode = params["drive_mode"]

            key = f"{name}.pos"
            norm = joints[key]

            if name == "gripper":
                # RANGE_0_100 reverse
                if drive_mode == 1:
                    norm = 100.0 - norm
                tick = (norm / 100.0) * (rng_max - rng_min) + rng_min
            else:
                # RANGE_M100_100 reverse
                if drive_mode == 1:
                    norm = -norm
                tick = ((norm + 100.0) / 200.0) * (rng_max - rng_min) + rng_min

            ticks[i] = round(tick)
        return ticks

    def _radians_to_normalized(self, radians: np.ndarray) -> dict[str, float]:
        ticks = self._radians_to_ticks(radians)
        return self._ticks_to_normalized(ticks)

    def _normalized_to_radians(self, joints: dict[str, float]) -> np.ndarray:
        ticks = self._normalized_to_ticks(joints)
        return self._ticks_to_radians(ticks)

    async def connect(self) -> None:
        logger.info(f"Connecting to SO101 {self._mode} on {self._robot.port}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_CONNECT):
                await asyncio.to_thread(self._robot.connect)

            if self._mode == "follower":
                self.is_controlled = True
            else:
                self.is_controlled = False
        except TimeoutError:
            logger.error("Timeout connecting to robot")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            raise

    async def disconnect(self) -> None:
        logger.info(f"Disconnecting SO101 {self._mode} on {self._robot.port}")
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                await asyncio.to_thread(self._robot.disconnect)
            logger.info("Robot disconnected")
        except TimeoutError:
            logger.warning("Timeout during robot disconnect - forcing cleanup")
        except Exception as e:
            logger.error(f"Error during robot disconnect: {e}")

    async def ping(self) -> dict:
        return self._create_event("pong")

    async def set_joints_state(self, joints: dict, goal_time: float) -> dict:
        await self._move_to_target(joints, goal_time)
        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        logger.info("Enabling torque")
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot._set_torque, enabled=True)
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        logger.info("Disabling torque")
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot._set_torque, enabled=False)
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            state = await self._get_state()
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        return self._create_event(
            "force_was_updated",
            state=None,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        raise NotImplementedError("Force control is not implemented for SO101")

    def features(self) -> list[str]:
        return [f"{name}.pos" for name in SO101_JOINT_ORDER]

    async def _get_state(self) -> dict[str, float]:
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)
        return self._radians_to_normalized(obs.joint_positions)

    async def _move_to_target(self, joints: dict, goal_time: float) -> None:
        max_rad = MAX_SPEED_RAD_S * goal_time

        # Read current state in radians
        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)
        current_rad = obs.joint_positions

        # Convert target from normalized to radians
        target_rad = self._normalized_to_radians(joints)

        # Clamp previous target blend in radian space
        if self.previous_target is not None:
            prev_rad = self._normalized_to_radians(self.previous_target)
            for i in range(len(current_rad)):
                current_rad[i] = current_rad[i] + _clamp(prev_rad[i] - current_rad[i], max_rad * 2)

        # Clamp velocity: limit per-joint movement to max_rad per cycle
        clamped_rad = np.array(
            [current_rad[i] + _clamp(target_rad[i] - current_rad[i], max_rad) for i in range(len(current_rad))],
            dtype=np.float32,
        )

        self.previous_target = self._radians_to_normalized(clamped_rad)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.send_action, clamped_rad)
