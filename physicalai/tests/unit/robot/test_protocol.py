# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Robot Protocol definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from physicalai.robot.interface import Robot, RobotObservation

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


@dataclass
class _Obs(RobotObservation):
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None


class _ValidRobot:
    """Minimal class satisfying the Robot protocol."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> _Obs:
        return _Obs(joint_positions=np.zeros(6, dtype=np.float32), timestamp=0.0)

    def send_action(self, action: np.ndarray) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1", "j2", "j3", "j4", "j5"]


class _MissingConnect:
    """Class missing the connect method."""

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> _Obs:
        return _Obs(joint_positions=np.zeros(6, dtype=np.float32), timestamp=0.0)

    def send_action(self, action: np.ndarray) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1", "j2", "j3", "j4", "j5"]


class _MissingSendAction:
    """Class missing the send_action method."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> _Obs:
        return _Obs(joint_positions=np.zeros(6, dtype=np.float32), timestamp=0.0)

    def is_connected(self) -> bool:
        return True

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1", "j2", "j3", "j4", "j5"]


class TestRobotProtocol:
    """Tests for the Robot Protocol (runtime_checkable)."""

    def test_valid_robot_is_instance(self) -> None:
        """A class with all required members satisfies the protocol."""
        robot = _ValidRobot()
        assert isinstance(robot, Robot)

    def test_missing_connect_not_instance(self) -> None:
        """A class missing connect() does not satisfy the protocol."""
        robot = _MissingConnect()
        assert not isinstance(robot, Robot)

    def test_missing_send_action_not_instance(self) -> None:
        """A class missing send_action() does not satisfy the protocol."""
        robot = _MissingSendAction()
        assert not isinstance(robot, Robot)
