# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WidowXAI protocol adapter.

Wraps physicalai's ``WidowXAI`` driver behind the backend's ``RobotClient``
interface with joint/unit conversion and async-safe hardware access.
"""

import asyncio
from typing import Literal

import numpy as np
from loguru import logger
from physicalai.robot.trossen import WidowXAI, WidowXAIObservation

from robots.robot_client import RobotClient
from schemas.robot import RobotType

HARDWARE_TIMEOUT_CONNECT = 10.0
HARDWARE_TIMEOUT_COMMAND = 5.0


class WidowXAIAdapter(RobotClient):
    """Adapt physicalai's :class:`WidowXAI` to the backend's RobotClient API.

    Read path converts radians to app-facing degrees (except gripper), and
    write path converts app-facing degrees back to radians.
    """

    name = "WidowXAI"

    def __init__(self, robot: WidowXAI, mode: Literal["follower", "leader"]) -> None:
        self._robot = robot
        self._mode = mode
        self._bus_lock = asyncio.Lock()
        self.is_controlled: bool = False

    @property
    def robot_type(self) -> RobotType:
        if self._mode == "follower":
            return RobotType.TROSSEN_WIDOWXAI_FOLLOWER
        return RobotType.TROSSEN_WIDOWXAI_LEADER

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected()

    def _observation_to_state(self, obs: WidowXAIObservation) -> dict[str, float]:
        result: dict[str, float] = {}
        sensor_data = obs.sensor_data
        if sensor_data is None:
            raise RuntimeError("Robot observation is missing sensor data")

        for i, name in enumerate(self._robot.joint_names):
            if name == "gripper":
                pos = float(obs.joint_positions[i])
            else:
                pos = float(np.rad2deg(obs.joint_positions[i]))

            vel = float(sensor_data["velocities"][i])
            result[f"{name}.pos"] = pos
            result[f"{name}.vel"] = vel

        return result

    def _state_to_action(self, joints: dict) -> np.ndarray:
        positions = np.zeros(len(self._robot.joint_names))

        for i, name in enumerate(self._robot.joint_names):
            if name == "gripper":
                positions[i] = joints[f"{name}.pos"]
            else:
                positions[i] = np.deg2rad(joints[f"{name}.pos"])

        return positions

    async def connect(self) -> None:
        logger.info(f"Connecting to WidowXAI {self._mode} at {self._robot.ip}")
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
        logger.info(f"Disconnecting WidowXAI {self._mode} at {self._robot.ip}")
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
        if self._mode == "leader":
            raise RuntimeError("Cannot send actions to a leader arm")

        positions = self._state_to_action(joints)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(
                self._robot.send_action,
                positions,
                # Increase default goal_time to reduce oscillations due to the application
                # performing teleoperation at 30Hz
                goal_time=3 * goal_time,
            )

        return self._create_event("joints_state_was_set", joints=joints)

    async def enable_torque(self) -> dict:
        self.is_controlled = True
        return self._create_event("torque_was_enabled")

    async def disable_torque(self) -> dict:
        self.is_controlled = False
        return self._create_event("torque_was_disabled")

    async def read_state(self, *, normalize: bool = True) -> dict:  # noqa: ARG002
        try:
            async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
                obs = await asyncio.to_thread(self._robot.get_observation)
            state = self._observation_to_state(obs)
            return self._create_event(
                "state_was_updated",
                state=state,
                is_controlled=self.is_controlled,
            )
        except Exception as e:
            logger.error(f"Robot read error: {e}")
            raise

    async def read_forces(self) -> dict | None:
        if self._mode == "leader":
            return None

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            obs = await asyncio.to_thread(self._robot.get_observation)

        sensor_data = obs.sensor_data
        if sensor_data is None:
            raise RuntimeError("Robot observation is missing sensor data")

        forces = {}
        for i, name in enumerate(self._robot.joint_names):
            forces[f"{name}.eff"] = float(sensor_data["efforts"][i])

        return self._create_event(
            "force_was_updated",
            state=forces,
            is_controlled=self.is_controlled,
        )

    async def set_forces(self, forces: dict) -> dict:
        if self._mode == "follower":
            logger.warning("Cannot send forces to a follower arm")
            return forces

        efforts = np.zeros(len(self._robot.joint_names))
        for i, name in enumerate(self._robot.joint_names):
            efforts[i] = forces.get(f"{name}.eff", 0.0)

        async with self._bus_lock, asyncio.timeout(HARDWARE_TIMEOUT_COMMAND):
            await asyncio.to_thread(self._robot.set_external_efforts, efforts, 0.1)

        return forces

    def features(self) -> list[str]:
        positions: list[str] = [f"{name}.pos" for name in self._robot.joint_names]
        velocities: list[str] = [f"{name}.vel" for name in self._robot.joint_names]
        return positions + velocities
