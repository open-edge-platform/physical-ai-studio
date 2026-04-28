# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bimanual Trossen WidowX AI robot driver.

Composes two :class:`WidowXAI` arms (left + right) behind the
:class:`~physicalai.robot.Robot` protocol, prefixing all joint names with
``left_`` / ``right_`` respectively.

Usage::

    from physicalai.robot.trossen import WidowXAI, BimanualWidowXAI

    left = WidowXAI(ip="192.168.1.10", role="follower")
    right = WidowXAI(ip="192.168.1.11", role="follower")
    robot = BimanualWidowXAI(left=left, right=right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from physicalai.robot import Robot

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.trossen.widowxai import WidowXAI


@dataclass
class BimanualWidowXAIObservation:
    """Merged observation from a bimanual WidowX AI robot.

    Attributes:
        joint_positions: Array of shape ``(14,)`` — left (7) then right (7)
            joint positions in radians.
        timestamp: ``time.monotonic()`` at the moment of capture (left arm).
        sensor_data: Merged velocities and (for follower) external efforts,
            each of shape ``(14,)``.
        images: Always ``None`` — no built-in camera support.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None


class BimanualWidowXAI(Robot):
    """Two-arm WidowX AI driver composing a left and right :class:`WidowXAI`.

    Args:
        left: Left arm driver.
        right: Right arm driver.

    Raises:
        ValueError: If the two arms have different roles.
    """

    def __init__(self, left: WidowXAI, right: WidowXAI) -> None:
        """Initialize BimanualWidowXAI with two single-arm drivers.

        Raises:
            ValueError: If left and right arms have different roles.
        """
        if left.role != right.role:
            msg = f"Both arms must have the same role; got left={left.role!r}, right={right.role!r}."
            raise ValueError(msg)
        if type(left) is not type(right):
            msg = (
                f"Both arms must be the same driver type; got left={type(left).__name__}, right={type(right).__name__}."
            )
            raise ValueError(msg)

        self._left = left
        self._right = right

    @property
    def role(self) -> str:
        """Robot role (``"leader"`` or ``"follower"``)."""
        return self._left.role

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names with ``left_`` / ``right_`` prefix."""
        left_names = [f"left_{n}" for n in self._left.joint_names]
        right_names = [f"right_{n}" for n in self._right.joint_names]
        return left_names + right_names

    def connect(self) -> None:
        """Connect both arms."""
        self._left.connect()
        try:
            self._right.connect()
        except Exception:
            self._left.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect both arms."""
        self._left.disconnect()
        self._right.disconnect()

    def is_connected(self) -> bool:
        """Return ``True`` when both arms are connected."""
        return self._left.is_connected() and self._right.is_connected()

    def get_observation(self) -> BimanualWidowXAIObservation:
        """Read and merge observations from both arms.

        Returns:
            Merged observation with left-then-right joint positions and
            sensor data arrays of shape ``(14,)``.
        """
        left_obs = self._left.get_observation()
        right_obs = self._right.get_observation()

        positions = np.concatenate([left_obs.joint_positions, right_obs.joint_positions])

        sensor_data: dict[str, np.ndarray] | None = None
        if left_obs.sensor_data is not None and right_obs.sensor_data is not None:
            sensor_data = {
                key: np.concatenate([left_value, right_obs.sensor_data[key]])
                for key, left_value in left_obs.sensor_data.items()
            }

        return BimanualWidowXAIObservation(
            joint_positions=positions,
            timestamp=left_obs.timestamp,
            sensor_data=sensor_data,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send a 14-DOF joint position command (follower only).

        Args:
            action: Array of shape ``(14,)`` — left (7) then right (7) target
                positions in radians.
            goal_time: Minimum time (seconds) for the arms to reach the target.

        Raises:
            RuntimeError: If called on a leader robot.
            ValueError: If action shape is not ``(14,)``.
        """
        if self.role == "leader":
            msg = "Cannot send actions to a leader robot."
            raise RuntimeError(msg)

        n = self._left.NUM_JOINTS
        expected_shape = (2 * n,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        self._left.send_action(action[:n], goal_time=goal_time)
        self._right.send_action(action[n:], goal_time=goal_time)

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None:
        """Apply force feedback (leader only).

        Args:
            efforts: Array of shape ``(14,)`` — left (7) then right (7) effort
                values.
            gain: Scaling factor for efforts. Defaults to 1.0.

        Raises:
            RuntimeError: If called on a follower robot.
            ValueError: If efforts shape is not ``(14,)``.
        """
        if self.role != "leader":
            msg = "set_external_efforts is only available for leader robots."
            raise RuntimeError(msg)

        n = self._left.NUM_JOINTS
        expected_shape = (2 * n,)
        if efforts.shape != expected_shape:
            msg = f"Expected efforts shape {expected_shape}, got {efforts.shape}"
            raise ValueError(msg)

        self._left.set_external_efforts(efforts[:n], gain)
        self._right.set_external_efforts(efforts[n:], gain)
