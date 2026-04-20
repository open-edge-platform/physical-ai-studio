# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trossen WidowX AI robot arm driver.

Concrete implementation of the :class:`~physicalai.robot.Robot`
protocol for the Trossen WidowX AI robot arm (7-DOF).

Requires the ``trossen-arm`` package::

    pip install physicalai[trossen]

The driver supports two roles:

* **follower** (default) — position control, used for inference / deployment.
* **leader** — external effort mode, used for teleoperation.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import trossen_arm
from loguru import logger

from physicalai.robot import Robot
from physicalai.robot.trossen.constants import HOME_POSITION, VALID_ROLES, WIDOWXAI_JOINT_ORDER

if TYPE_CHECKING:
    from trossen_arm import TrossenArmDriver

    from physicalai.capture.frame import Frame


@dataclass
class WidowXAIObservation:
    """Observation from the WidowX AI robot arm.

    Attributes:
        joint_positions: Array of shape ``(7,)`` with joint positions in radians.
        timestamp: ``time.monotonic()`` at the moment of capture.
        sensor_data: Velocities and (for follower) external efforts.
        images: Always ``None`` — no built-in camera support.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None


class WidowXAI(Robot):
    """Driver for the Trossen WidowX AI robot arm (7-DOF).

    Args:
        ip: IP address of the robot arm (e.g. ``"192.168.1.2"``).
        role: ``"follower"`` (position control) or ``"leader"``
            (external effort mode for teleoperation).
    """

    JOINT_ORDER: ClassVar[list[str]] = list(WIDOWXAI_JOINT_ORDER)
    NUM_JOINTS: ClassVar[int] = len(JOINT_ORDER)

    def __init__(self, ip: str, role: Literal["leader", "follower"] = "follower") -> None:
        """Initialize the WidowXAI driver.

        Raises:
            ValueError: If role is not "leader" or "follower".
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. Must be one of {VALID_ROLES}."
            raise ValueError(msg)

        self._ip = ip
        self._role = role
        self._driver: TrossenArmDriver | None = None
        self._last_positions: np.ndarray | None = None

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names matching state/action arrays."""
        return self.JOINT_ORDER

    @property
    def ip(self) -> str:
        """IP address of the robot arm."""
        return self._ip

    @property
    def role(self) -> str:
        """Robot role (``"leader"`` or ``"follower"``)."""
        return self._role

    def _require_driver(self) -> TrossenArmDriver:
        if self._driver is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return self._driver

    def connect(self) -> None:
        """Connect to the robot and move to home position."""
        if self.is_connected():
            return

        driver = trossen_arm.TrossenArmDriver()

        try:
            end_effector = (
                trossen_arm.StandardEndEffector.wxai_v0_follower
                if self._role == "follower"
                else trossen_arm.StandardEndEffector.wxai_v0_leader
            )
            driver.configure(
                trossen_arm.Model.wxai_v0,
                end_effector,
                self._ip,
                True,  # noqa: FBT003
                timeout=5,
            )

            if self._role == "leader":
                # Zero external efforts before homing (essential for leader)
                driver.set_all_modes(trossen_arm.Mode.external_effort)
                driver.set_all_external_efforts(list(HOME_POSITION), 0.0, False)  # noqa: FBT003

            driver.set_all_modes(trossen_arm.Mode.position)
            driver.set_all_positions(list(HOME_POSITION), 2.0, True)  # noqa: FBT003

        except Exception:
            with contextlib.suppress(Exception):
                driver.cleanup()
            self._driver = None
            raise

        self._driver = driver
        logger.info(f"WidowXAI connected at {self._ip} (role={self._role})")

    def disconnect(self) -> None:
        """Home the arm and clean up the SDK connection."""
        if self._driver is None:
            logger.warning(f"WidowXAI not connected at {self._ip} (role={self._role})")
            return

        try:
            self._driver.set_all_modes(trossen_arm.Mode.position)
            self._driver.set_all_positions(list(HOME_POSITION), 2.0, True)  # noqa: FBT003
        except Exception:  # noqa: BLE001
            logger.warning("Failed to home WidowXAI during disconnect; proceeding to cleanup.")
        finally:
            try:
                self._driver.cleanup()
            except Exception:  # noqa: BLE001
                logger.warning("Error during WidowXAI cleanup; continuing.")
            self._driver = None
            self._last_positions = None

    def get_observation(self) -> WidowXAIObservation:
        """Read current joint state and auxiliary sensor data.

        Returns:
            Observation containing joint positions, timestamp, and sensor data.
        """
        driver = self._require_driver()

        positions = driver.get_all_positions()
        velocities = driver.get_all_velocities()

        positions_array = np.array(positions, dtype=np.float32)
        velocities_array = np.array(velocities, dtype=np.float32)

        sensor_data: dict[str, np.ndarray] = {"velocities": velocities_array}

        if self._role == "follower":
            efforts = driver.get_all_external_efforts()
            sensor_data["efforts"] = np.array(efforts, dtype=np.float32)

        return WidowXAIObservation(
            joint_positions=positions_array,
            timestamp=time.monotonic(),
            sensor_data=sensor_data,
        )

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Send a 7-DOF joint position command to follower arms.

        Args:
            action: Array of shape ``(7,)`` with target joint positions in radians.
            goal_time: Minimum time (seconds) for the arm to reach the target.
                The backend control loop typically sets this to ``1 / fps``.
                Not part of the :class:`~physicalai.robot.Robot` protocol — the
                adapter passes it explicitly when it knows the concrete type.

        Raises:
            RuntimeError: If called on a leader arm.
            ValueError: If action shape is not ``(7,)``.
        """
        if self._role == "leader":
            msg = "Cannot send actions to a leader arm."
            raise RuntimeError(msg)

        expected_shape = (self.NUM_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        driver = self._require_driver()

        if self._last_positions is not None:
            velocities = ((action - self._last_positions) / goal_time).tolist()
            driver.set_all_positions(action.tolist(), goal_time, False, velocities)  # noqa: FBT003
        else:
            driver.set_all_positions(action.tolist(), goal_time, False)  # noqa: FBT003

        self._last_positions = action.copy()

    def is_connected(self) -> bool:
        """Return True when the SDK driver is configured."""
        return self._driver is not None and self._driver.get_is_configured()

    def set_external_efforts(self, efforts: np.ndarray, gain: float = 1.0) -> None:
        """Apply force feedback (leader only).

        Switches to external_effort mode and applies negated efforts,
        matching the backend's force feedback convention.

        Args:
            efforts: Array of shape ``(7,)`` with effort values.
            gain: Scaling factor for efforts. Defaults to 1.0.

        Raises:
            RuntimeError: If called on a follower arm.
            ValueError: If efforts shape is not ``(7,)``.
        """
        if self._role != "leader":
            msg = "set_external_efforts is only available for leader arms."
            raise RuntimeError(msg)

        expected_shape = (self.NUM_JOINTS,)
        if efforts.shape != expected_shape:
            msg = f"Expected efforts shape {expected_shape}, got {efforts.shape}"
            raise ValueError(msg)

        driver = self._require_driver()

        driver.set_all_modes(trossen_arm.Mode.external_effort)
        driver.set_all_external_efforts([-gain * e for e in efforts.tolist()], 0.0, False)  # noqa: FBT003
