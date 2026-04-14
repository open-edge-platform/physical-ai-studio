# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot Protocol-based interface definition.

Defines the structural interface that all robot implementations must satisfy.
Uses Python's Protocol for structural (duck) typing — no inheritance required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from physicalai.capture.frame import Frame


@runtime_checkable
class RobotObservation(Protocol):
    """Observation from robot hardware.

    Returned by :meth:`Robot.get_observation`. Third-party implementations
    may use any concrete class (dataclass, NamedTuple, etc.) as long as it
    exposes these attributes.

    Attributes:
        joint_positions: Array of shape ``(N,)`` with joint positions, ordered
            to match :attr:`Robot.joint_names`.
        timestamp: ``time.monotonic()`` at the moment of capture.
        sensor_data: Optional auxiliary sensor readings keyed by name
            (e.g. ``{"imu": np.array([...])}``). ``None`` if no extra
            sensors are available.
        images: Optional built-in camera frames keyed by camera name.
            ``None`` if the robot has no embedded cameras. External cameras
            are managed separately via ``physicalai.capture``.
    """

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None
    images: dict[str, Frame] | None


@runtime_checkable
class Robot(Protocol):
    """Structural interface for robot implementations.

    Any class that implements these methods is a valid robot.
    No inheritance required. No registration required.

    Example:
        >>> class MyRobot:
        ...     joint_names = ["shoulder", "elbow", "wrist"]
        ...     def connect(self) -> None: ...
        ...     def disconnect(self) -> None: ...
        ...     def get_observation(self) -> MyObservation: ...
        ...     def send_action(self, action: np.ndarray) -> None: ...
        ...     def is_connected(self) -> bool: ...
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
        is closed. This method is called automatically by the ``connect()``
        context manager, including when exceptions occur.
        """
        ...

    def get_observation(self) -> RobotObservation:
        """Read the current robot state.

        Returns:
            A :class:`RobotObservation` containing at minimum:

            - ``joint_positions``: ``np.ndarray`` of shape ``(N,)`` with
              current joint positions, ordered to match :attr:`joint_names`.
            - ``timestamp``: ``float`` from ``time.monotonic()``.
            - ``sensor_data``: optional dict of auxiliary sensor arrays.
            - ``images``: optional dict of built-in camera frames.
        """
        ...

    def send_action(self, action: np.ndarray) -> None:
        """Send a joint command to the robot.

        Args:
            action: Array of shape ``(N,)`` matching :attr:`joint_names`.
                The semantics (positions, velocities, torques) depend on the
                robot implementation. For teleoperation, this is typically
                ``leader.get_observation().joint_positions`` passed directly.
        """
        ...

    def is_connected(self) -> bool:
        """Check if the robot is currently connected.

        Returns:
            True if the robot is connected and ready to receive commands.
            False if not connected or in an error state.
        """
        ...

    @property
    def joint_names(self) -> list[str]:
        """Ordered joint names.

        The length and order must match the ``joint_positions`` array returned
        by :meth:`get_observation` and the ``action`` array accepted by
        :meth:`send_action`.
        """
        ...
