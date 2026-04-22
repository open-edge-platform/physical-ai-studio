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
        read:  physicalai (radians) → normalized
        write: normalized → radians
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
        self._bus_lock = asyncio.Lock()

        self.previous_target: dict[str, float] | None = None
        self.is_controlled: bool = False

        self._joint_params: dict[str, dict] = {}
        for name in SO101_JOINT_ORDER:
            cal_val = calibration.values[name]
            direction = -1 if cal_val.drive_mode == 1 else 1
            rng = cal_val.range_max - cal_val.range_min
            is_gripper = name == "gripper"

            # Precompute linear coefficients: norm = rad * scale + bias
            # Derived from combining tick<->radian and tick<->normalized formulas:
            #   tick = rad / (direction * RADIANS_PER_TICK) + homing_offset
            #   norm_body = ((tick - min) / range) * 200 - 100
            #   norm_grip = ((tick - min) / range) * 100
            if rng == 0:
                scale = 0.0
                bias = 0.0
            elif is_gripper:
                scale = 100.0 / (direction * RADIANS_PER_TICK * rng)
                bias = ((cal_val.homing_offset - cal_val.range_min) / rng) * 100.0
            else:
                scale = 200.0 / (direction * RADIANS_PER_TICK * rng)
                bias = ((cal_val.homing_offset - cal_val.range_min) / rng) * 200.0 - 100.0

            # drive_mode=1 flips: body → negate, gripper → 100-norm
            if cal_val.drive_mode == 1 and not is_gripper:
                scale = -scale
                bias = -bias
            elif cal_val.drive_mode == 1 and is_gripper:
                scale = -scale
                bias = 100.0 - bias

            self._joint_params[name] = {
                "scale": scale,
                "bias": bias,
                "is_gripper": is_gripper,
            }

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.SO101_FOLLOWER
        return RobotType.SO101_LEADER

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    # Linear conversion between physicalai radians and backend normalized values.
    # Body joints use [-100, 100], gripper uses [0, 100].
    # Coefficients (scale, bias) are precomputed per joint in __init__.

    def _radians_to_normalized(self, radians: np.ndarray) -> dict[str, float]:
        result: dict[str, float] = {}
        for i, name in enumerate(SO101_JOINT_ORDER):
            p = self._joint_params[name]
            norm = float(radians[i]) * p["scale"] + p["bias"]
            result[f"{name}.pos"] = max(0.0, min(100.0, norm)) if p["is_gripper"] else max(-100.0, min(100.0, norm))
        return result

    def _normalized_to_radians(self, joints: dict[str, float]) -> np.ndarray:
        result = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
        for i, name in enumerate(SO101_JOINT_ORDER):
            p = self._joint_params[name]
            result[i] = (joints[f"{name}.pos"] - p["bias"]) / p["scale"] if p["scale"] != 0 else 0.0
        return result

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
