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


def _validate_sensor_data(sensor_data: object) -> None:
    """Validate optional sensor_data payload.

    Raises:
        RobotVerificationError: If sensor_data is not a dict of np.ndarrays.
    """
    if sensor_data is None:
        return
    if not isinstance(sensor_data, dict):
        msg = "sensor_data must be a dict"
        raise RobotVerificationError(msg)
    for name, values in sensor_data.items():
        if not isinstance(values, np.ndarray):
            msg = f"sensor_data['{name}'] must be np.ndarray"
            raise RobotVerificationError(msg)
    logger.success("Check 3 passed: sensor_data contains valid arrays")


def _validate_image_frame(name: str, frame: object) -> None:
    """Validate one frame-like image object.

    Raises:
        RobotVerificationError: If frame lacks required attributes or types.
    """
    if not hasattr(frame, "data"):
        msg = f"image '{name}' must contain frame.data"
        raise RobotVerificationError(msg)
    if not hasattr(frame, "timestamp"):
        msg = f"image '{name}' must contain frame.timestamp"
        raise RobotVerificationError(msg)
    if not hasattr(frame, "sequence"):
        msg = f"image '{name}' must contain frame.sequence"
        raise RobotVerificationError(msg)

    img = frame.data
    if not isinstance(img, np.ndarray):
        msg = f"image '{name}' frame.data must be np.ndarray"
        raise RobotVerificationError(msg)
    if img.ndim != 3:  # noqa: PLR2004
        msg = f"image '{name}' frame.data must be 3D (H, W, C), got ndim={img.ndim}"
        raise RobotVerificationError(msg)
    if not isinstance(frame.timestamp, (int, float)):
        msg = f"image '{name}' frame.timestamp must be numeric"
        raise RobotVerificationError(msg)
    if not isinstance(frame.sequence, int):
        msg = f"image '{name}' frame.sequence must be int"
        raise RobotVerificationError(msg)

    logger.debug("image '{}': shape={}, dtype={}", name, img.shape, img.dtype)


def _validate_images(images: object) -> None:
    """Validate optional image payload.

    Raises:
        RobotVerificationError: If images is not a dict of valid frame objects.
    """
    if images is None:
        return
    if not isinstance(images, dict):
        msg = "images must be a dict"
        raise RobotVerificationError(msg)
    for name, frame in images.items():
        _validate_image_frame(name, frame)
    logger.success("Check 4 passed: images dict contains valid frames")


def _validate_observation(obs: object) -> np.ndarray:
    """Validate observation structure and return joint positions.

    Returns:
        The validated joint_positions array.

    Raises:
        RobotVerificationError: If observation lacks required attributes or types.
    """
    if not hasattr(obs, "joint_positions"):
        msg = "observation must contain 'joint_positions'"
        raise RobotVerificationError(msg)
    joint_positions = obs.joint_positions
    if not isinstance(joint_positions, np.ndarray):
        msg = "joint_positions must be np.ndarray"
        raise RobotVerificationError(msg)
    logger.debug("joint_positions: shape={}, dtype={}", joint_positions.shape, joint_positions.dtype)

    if not hasattr(obs, "timestamp"):
        msg = "observation must contain 'timestamp'"
        raise RobotVerificationError(msg)
    if not isinstance(obs.timestamp, (int, float)):
        msg = "timestamp must be numeric"
        raise RobotVerificationError(msg)
    logger.debug("timestamp: {}", obs.timestamp)
    logger.success("Check 2 passed: observation contains valid 'joint_positions' and 'timestamp'")

    _validate_sensor_data(getattr(obs, "sensor_data", None))
    _validate_images(getattr(obs, "images", None))
    return joint_positions


def _verify_stationary_after_reconnect(robot: Robot) -> None:
    """Verify robot state is stationary after reconnect.

    Raises:
        RobotVerificationError: If joints move after disconnect/reconnect.
    """
    logger.debug("Testing disconnect() -> connect() lifecycle")
    robot.disconnect()
    robot.connect()

    obs1 = robot.get_observation()
    time.sleep(0.1)
    obs2 = robot.get_observation()

    if not np.allclose(obs1.joint_positions, obs2.joint_positions, atol=0.01):
        msg = (
            f"Robot must be stationary after disconnect(). State changed from {obs1.joint_positions} to "
            f"{obs2.joint_positions}"
        )
        raise RobotVerificationError(msg)
    logger.success("Check 6 passed: robot is stationary after disconnect() -> connect()")


def verify_robot(robot: Robot, _num_steps: int = 10) -> None:
    """Verify a robot implementation satisfies the Protocol contract.

    Runs a sequence of checks against a *real* (or sufficiently realistic mock)
    robot instance.  The robot must **not** be connected when this function is
    called - the function manages the full lifecycle itself.

     Checks:
          1. ``connect()`` / ``disconnect()`` lifecycle.
          2. ``get_observation()`` returns valid ``joint_positions`` (np.ndarray)
              and ``timestamp`` (numeric).
          3. If ``sensor_data`` is present, it must be a dict of np.ndarrays.
          4. If ``images`` is present, it must be a dict of frame-like objects
              with ``data`` (3-D np.ndarray), ``timestamp`` (numeric), and
              ``sequence`` (int).
          5. ``send_action()`` accepts a numpy array shaped like
              ``joint_positions``.
          6. After ``disconnect()`` -> ``connect()``, the robot should be
              stationary (joint positions unchanged within tolerance over 0.1 s).

    Args:
        robot: An object that is expected to satisfy the Robot protocol.
        _num_steps: Number of observation/action round-trips to execute
            (currently reserved for future use).
    """
    logger.info("Verifying robot: {}", repr(robot))

    with connect(robot):
        logger.success("Check 1 passed: connect() lifecycle")
        joint_positions = _validate_observation(robot.get_observation())

        # Echo joint positions as an action to verify send_action() accepts them.
        robot.send_action(joint_positions.copy())
        logger.success("Check 5 passed: send_action() accepted joint_positions-shaped action")

        _verify_stationary_after_reconnect(robot)

    logger.success("All checks passed for robot: {}", repr(robot))
