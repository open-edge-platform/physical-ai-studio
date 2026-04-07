# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot verification utilities.

Provides ``verify_robot()`` for verifying that a robot implementation
satisfies the :class:`~physicalai.robot.interface.Robot` protocol contract.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from physicalai.robot.connect import connect

if TYPE_CHECKING:
    from physicalai.robot.interface import Robot


class RobotVerificationError(ValueError):
    """Raised when a robot implementation violates the protocol contract."""

    def __init__(self, message: str) -> None:
        """Initialize the error and emit a verification failure log."""
        logger.error("Verification failed: {}", message)
        super().__init__(message)


def verify_robot(robot: Robot, _num_steps: int = 10) -> None:
    """Verify a robot implementation satisfies the Protocol contract.

    Runs a sequence of checks against a *real* (or sufficiently realistic mock)
    robot instance.  The robot must **not** be connected when this function is
    called - the function manages the full lifecycle itself.

    Checks:
        1. ``connect()`` / ``disconnect()`` lifecycle.
        2. ``get_observation()`` returns a dict with ``"state"`` (np.ndarray)
           and ``"timestamp"`` (numeric).
        3. If ``"images"`` is present, it must be a dict of 3-D np.ndarrays.
        4. ``send_action()`` accepts a numpy array shaped like the state.
        5. After ``disconnect()`` -> ``connect()``, the robot should be
           stationary (state unchanged within tolerance over 0.1 s).

    Args:
        robot: An object that is expected to satisfy the Robot protocol.
        _num_steps: Number of observation/action round-trips to execute
            (currently reserved for future use).

    Raises:
        RobotVerificationError: If any protocol contract check fails.
    """
    logger.info("Verifying robot: {}", repr(robot))

    with connect(robot):
        logger.success("Check 1 passed: connect() lifecycle")

        obs = robot.get_observation()
        if not isinstance(obs, dict):
            msg = "get_observation() must return a dict"
            raise RobotVerificationError(msg)
        logger.debug("get_observation() returned dict")

        if "state" not in obs:
            msg = "observation must contain 'state'"
            raise RobotVerificationError(msg)
        if not isinstance(obs["state"], np.ndarray):
            msg = "state must be np.ndarray"
            raise RobotVerificationError(msg)
        logger.debug("state: shape={}, dtype={}", obs["state"].shape, obs["state"].dtype)

        if "timestamp" not in obs:
            msg = "observation must contain 'timestamp'"
            raise RobotVerificationError(msg)
        if not isinstance(obs["timestamp"], (int, float)):
            msg = "timestamp must be numeric"
            raise RobotVerificationError(msg)
        logger.debug("timestamp: {}", obs["timestamp"])
        logger.success("Check 2 passed: observation contains valid 'state' and 'timestamp'")

        if "images" in obs:
            if not isinstance(obs["images"], dict):
                msg = "images must be a dict"
                raise RobotVerificationError(msg)
            for name, img in obs["images"].items():
                if not isinstance(img, np.ndarray):
                    msg = f"image '{name}' must be np.ndarray"
                    raise RobotVerificationError(msg)
                if img.ndim != 3:  # noqa: PLR2004
                    msg = f"image '{name}' must be 3D (C, H, W), got ndim={img.ndim}"
                    raise RobotVerificationError(msg)
                logger.debug("image '{}': shape={}, dtype={}", name, img.shape, img.dtype)
            logger.success("Check 3 passed: images dict contains valid 3D arrays")

        # Echo the state back as an action to verify send_action() accepts it without error
        action = obs["state"].copy()
        robot.send_action(action)
        logger.success("Check 4 passed: send_action() accepted state-shaped action")

        logger.debug("Testing disconnect() -> connect() lifecycle")
        robot.disconnect()
        robot.connect()

        obs1 = robot.get_observation()
        time.sleep(0.1)
        obs2 = robot.get_observation()

        if not np.allclose(obs1["state"], obs2["state"], atol=0.01):
            msg = f"Robot must be stationary after disconnect(). State changed from {obs1['state']} to {obs2['state']}"
            raise RobotVerificationError(msg)
        logger.success("Check 5 passed: robot is stationary after disconnect() -> connect()")

    logger.success("All checks passed for robot: {}", repr(robot))
