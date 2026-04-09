# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Robot Protocol definition."""

from __future__ import annotations

from typing import Any

import numpy as np

from physicalai.robot.interface import Robot


class _ValidRobot:
    """Minimal class satisfying the Robot protocol."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"state": np.zeros(6, dtype=np.float32), "timestamp": 0.0}

    def send_action(self, action: np.ndarray) -> None:
        pass


class _MissingConnect:
    """Class missing the connect method."""

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"state": np.zeros(6, dtype=np.float32), "timestamp": 0.0}

    def send_action(self, action: np.ndarray) -> None:
        pass


class _MissingSendAction:
    """Class missing the send_action method."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"state": np.zeros(6, dtype=np.float32), "timestamp": 0.0}


class TestRobotProtocol:
    """Tests for the Robot Protocol (runtime_checkable)."""

    def test_valid_robot_is_instance(self) -> None:
        """A class with all four methods satisfies the protocol."""
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
