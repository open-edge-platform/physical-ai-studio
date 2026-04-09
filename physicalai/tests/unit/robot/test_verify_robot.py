# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for verify_robot()."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from physicalai.robot.verify import RobotVerificationError, verify_robot


class _ConformantRobot:
    """A fake robot that passes all conformance checks."""

    def __init__(self) -> None:
        self._state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        return {
            "state": self._state.copy(),
            "timestamp": 123.456,
        }

    def send_action(self, action: np.ndarray) -> None:
        pass


class _NonStationaryRobot(_ConformantRobot):
    """A robot that drifts after disconnect — should fail the stationarity check."""

    def __init__(self) -> None:
        super().__init__()
        self._call_count = 0

    def get_observation(self) -> dict[str, Any]:
        self._call_count += 1
        drift = np.ones(6, dtype=np.float32) * self._call_count * 0.1
        return {
            "state": self._state + drift,
            "timestamp": 123.456,
        }


class _MissingStateRobot:
    """A robot whose observation is missing the 'state' key."""

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"timestamp": 0.0}

    def send_action(self, action: np.ndarray) -> None:
        pass


class TestVerifyRobot:
    """Tests for verify_robot()."""

    def test_conformant_robot_passes(self) -> None:
        """A conformant robot passes all checks."""
        robot = _ConformantRobot()
        verify_robot(robot)

    def test_missing_state_fails(self) -> None:
        """A robot missing 'state' in observation fails."""
        robot = _MissingStateRobot()
        with pytest.raises(RobotVerificationError, match="observation must contain 'state'"):
            verify_robot(robot)

    def test_non_stationary_robot_fails(self) -> None:
        """A robot that drifts after disconnect fails stationarity check."""
        robot = _NonStationaryRobot()
        with pytest.raises(RobotVerificationError, match="Robot must be stationary"):
            verify_robot(robot)
