# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot Protocol-based interface definition.

Defines the structural interface that all robot implementations must satisfy.
Uses Python's Protocol for structural (duck) typing — no inheritance required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class Robot(Protocol):
    """Structural interface for robot implementations.

    Any class that implements these four methods is a valid robot.
    No inheritance required. No registration required.

    Example:
        >>> class MyRobot:
        ...     def connect(self) -> None: ...
        ...     def disconnect(self) -> None: ...
        ...     def get_observation(self) -> dict[str, Any]: ...
        ...     def send_action(self, action: np.ndarray) -> None: ...
        ...
        >>> isinstance(MyRobot(), Robot)
        True
    """

    def connect(self) -> None:
        """Establish connection to the robot hardware.

        Called once before the inference loop begins. Must be idempotent —
        calling connect() on an already-connected robot should be a no-op
        or raise a clear error.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the robot.

        Implementations MUST leave the robot in a safe, stationary state.
        Motors must be stopped or holding position before the connection
        is closed. This method is called automatically by the connect()
        context manager, including when exceptions occur.
        """
        ...

    def get_observation(self) -> dict[str, Any]:
        """Read the current robot state.

        Returns:
            A dict with the following conventional structure::

                {
                    "state": np.ndarray,    # joint positions, gripper, etc.
                    "timestamp": float,     # time.monotonic() or equivalent
                }

            The exact keys and shapes must match what the policy expects,
            as declared in the policy's manifest.json under io.inputs.

            Note: cameras are managed separately from the robot interface.
            See the Cameras section of the design docs for how to combine
            robot state and camera frames into a full observation.
        """
        ...

    def send_action(self, action: np.ndarray) -> None:
        """Send an action command to the robot.

        Args:
            action: A numpy array of joint commands. The shape and semantics
                (positions, velocities, torques) depend on the policy
                that produced the action. The robot implementation is
                responsible for interpreting them correctly.
        """
        ...
