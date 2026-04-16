# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the connect() context manager."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from physicalai.robot.connect import connect


@dataclass
class _Obs:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, object] | None = None


def _make_mock_robot() -> MagicMock:
    """Create a MagicMock that looks like a Robot."""
    robot = MagicMock()
    robot.get_observation.return_value = _Obs(joint_positions=np.zeros(6, dtype=np.float32), timestamp=0.0)
    return robot


class TestConnect:
    """Tests for the connect() context manager."""

    def test_connect_disconnect_called(self) -> None:
        """connect() calls robot.connect() on entry and robot.disconnect() on exit."""
        robot = _make_mock_robot()

        with connect(robot) as r:
            assert r is robot
            robot.connect.assert_called_once()
            robot.disconnect.assert_not_called()

        robot.disconnect.assert_called_once()

    def test_disconnect_called_on_exception(self) -> None:
        """robot.disconnect() is called even when an exception occurs."""
        robot = _make_mock_robot()

        with pytest.raises(RuntimeError, match="boom"):
            with connect(robot):
                raise RuntimeError("boom")

        robot.connect.assert_called_once()
        robot.disconnect.assert_called_once()

    def test_yields_robot(self) -> None:
        """The context manager yields the robot instance."""
        robot = _make_mock_robot()

        with connect(robot) as r:
            obs = r.get_observation()
            assert isinstance(obs.joint_positions, np.ndarray)
