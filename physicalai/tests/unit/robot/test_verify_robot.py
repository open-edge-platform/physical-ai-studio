# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify_robot()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from physicalai.robot.interface import RobotObservation
from physicalai.robot.verify import RobotVerificationError, verify_robot

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame


@dataclass
class _Obs(RobotObservation):
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None


class _ConformantRobot:
    """A fake robot that passes all conformance checks."""

    def __init__(self) -> None:
        self._state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> _Obs:
        return _Obs(joint_positions=self._state.copy(), timestamp=123.456)

    def send_action(self, action: np.ndarray) -> None:
        pass

    def is_connected(self) -> bool:
        return self._connected

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1", "j2", "j3", "j4", "j5"]


class _NonStationaryRobot(_ConformantRobot):
    """A robot that drifts after disconnect — should fail the stationarity check."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def get_observation(self) -> _Obs:
        self._call_count += 1
        drift = np.ones(6, dtype=np.float32) * self._call_count * 0.1
        return _Obs(joint_positions=self._state + drift, timestamp=123.456)


@dataclass
class _InvalidObs:
    """Observation with invalid joint_positions type for negative tests."""

    joint_positions: list[float]
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, object] | None = None


class _InvalidJointPositionsRobot:
    """A robot whose observation has invalid joint_positions type."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> _InvalidObs:
        return _InvalidObs(joint_positions=[0.0, 1.0], timestamp=0.0)

    def send_action(self, action: np.ndarray) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    @property
    def joint_names(self) -> list[str]:
        return ["j0", "j1", "j2", "j3", "j4", "j5"]


class TestVerifyRobot:
    """Tests for verify_robot()."""

    def test_conformant_robot_passes(self) -> None:
        """A conformant robot passes all checks."""
        robot = _ConformantRobot()
        verify_robot(robot)

    def test_invalid_joint_positions_type_fails(self) -> None:
        """A robot with non-array joint_positions fails."""
        robot = _InvalidJointPositionsRobot()
        with pytest.raises(RobotVerificationError, match="joint_positions must be np.ndarray"):
            verify_robot(cast(Any, robot))

    def test_non_stationary_robot_fails(self) -> None:
        """A robot that drifts after disconnect fails stationarity check."""
        robot = _NonStationaryRobot()
        with pytest.raises(RobotVerificationError, match="Robot must be stationary"):
            verify_robot(robot)
