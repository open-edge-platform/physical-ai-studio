# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the BimanualWidowXAI robot driver.

All hardware communication is mocked — no real hardware required.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock trossen_arm SDK (same pattern as test_widowxai.py)
# ---------------------------------------------------------------------------


def _make_mock_trossen_arm() -> MagicMock:
    module = MagicMock()

    driver = MagicMock()
    driver.get_is_configured.return_value = True
    driver.get_all_positions.return_value = [0.0] * 7
    driver.get_all_velocities.return_value = [0.0] * 7
    driver.get_all_external_efforts.return_value = [0.0] * 7
    driver.configure.return_value = None
    driver.cleanup.return_value = None
    driver.set_all_positions.return_value = None
    driver.set_all_modes.return_value = None
    driver.set_all_external_efforts.return_value = None

    module.TrossenArmDriver.return_value = driver
    module.Model.wxai_v0 = MagicMock(name="Model.wxai_v0")
    module.StandardEndEffector.wxai_v0_follower = MagicMock(name="wxai_v0_follower")
    module.StandardEndEffector.wxai_v0_leader = MagicMock(name="wxai_v0_leader")
    module.Mode.position = MagicMock(name="Mode.position")
    module.Mode.external_effort = MagicMock(name="Mode.external_effort")

    return module


@pytest.fixture
def mock_trossen_arm() -> Generator[MagicMock, None, None]:
    mock_module = _make_mock_trossen_arm()
    with (
        patch.dict("sys.modules", {"trossen_arm": mock_module}),
        patch("physicalai.robot.trossen.widowxai.trossen_arm", mock_module),
    ):
        yield mock_module


def _make_arm(mock_module: MagicMock, role: str = "follower", ip: str = "192.168.1.1") -> object:
    """Return a fresh WidowXAI with independent mock driver."""
    from physicalai.robot.trossen import WidowXAI

    # Give each arm its own driver mock so they don't share state
    arm_driver = MagicMock()
    arm_driver.get_is_configured.return_value = True
    arm_driver.get_all_positions.return_value = [0.0] * 7
    arm_driver.get_all_velocities.return_value = [0.0] * 7
    arm_driver.get_all_external_efforts.return_value = [0.0] * 7
    arm_driver.configure.return_value = None
    arm_driver.cleanup.return_value = None
    arm_driver.set_all_positions.return_value = None
    arm_driver.set_all_modes.return_value = None
    arm_driver.set_all_external_efforts.return_value = None

    mock_module.TrossenArmDriver.return_value = arm_driver

    robot = WidowXAI(ip=ip, role=role)  # type: ignore[arg-type]
    robot.connect()
    return robot


def _make_bimanual(mock_module: MagicMock, role: str = "follower"):
    from physicalai.robot.trossen.bimanual_widowxai import BimanualWidowXAI

    left = _make_arm(mock_module, role=role, ip="192.168.1.10")
    right = _make_arm(mock_module, role=role, ip="192.168.1.11")
    return BimanualWidowXAI(left=left, right=right)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestBimanualWidowXAIConstruction:
    def test_mismatched_roles_raise(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot.trossen import WidowXAI
        from physicalai.robot.trossen.bimanual_widowxai import BimanualWidowXAI

        left = WidowXAI(ip="192.168.1.10", role="follower")  # type: ignore[arg-type]
        right = WidowXAI(ip="192.168.1.11", role="leader")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="same role"):
            BimanualWidowXAI(left=left, right=right)  # type: ignore[arg-type]

    def test_joint_names_prefixed(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm)
        names = robot.joint_names
        assert len(names) == 14
        assert all(n.startswith("left_") for n in names[:7])
        assert all(n.startswith("right_") for n in names[7:])

    def test_role_reflects_arms(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")
        assert robot.role == "follower"

        robot_leader = _make_bimanual(mock_trossen_arm, role="leader")
        assert robot_leader.role == "leader"


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


class TestBimanualWidowXAIConnectivity:
    def test_is_connected_true_when_both_connected(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm)
        assert robot.is_connected() is True

    def test_is_connected_false_when_one_not(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm)
        # Forcibly clear one arm's driver
        robot._left._driver = None  # type: ignore[attr-defined]
        assert robot.is_connected() is False

    def test_connect_rolls_back_left_if_right_fails(self, mock_trossen_arm: MagicMock) -> None:
        from physicalai.robot.trossen import WidowXAI
        from physicalai.robot.trossen.bimanual_widowxai import BimanualWidowXAI

        left = WidowXAI(ip="192.168.1.10", role="follower")  # type: ignore[arg-type]
        right = WidowXAI(ip="192.168.1.11", role="follower")  # type: ignore[arg-type]

        bimanual = BimanualWidowXAI(left=left, right=right)

        left.connect = MagicMock()
        left.disconnect = MagicMock()
        right.connect = MagicMock(side_effect=RuntimeError("right arm failed"))

        with pytest.raises(RuntimeError, match="right arm failed"):
            bimanual.connect()

        left.connect.assert_called_once()
        right.connect.assert_called_once()
        left.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestBimanualWidowXAIObservation:
    def test_follower_observation_shape(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")
        obs = robot.get_observation()

        assert obs.joint_positions.shape == (14,)
        assert obs.sensor_data is not None
        assert obs.sensor_data["velocities"].shape == (14,)
        assert obs.sensor_data["efforts"].shape == (14,)

    def test_leader_observation_no_efforts(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="leader")
        obs = robot.get_observation()

        assert obs.joint_positions.shape == (14,)
        assert obs.sensor_data is not None
        assert "velocities" in obs.sensor_data
        assert "efforts" not in obs.sensor_data

    def test_observation_merges_left_right(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")

        left_pos = [float(i) for i in range(7)]
        right_pos = [float(i + 10) for i in range(7)]
        robot._left._driver.get_all_positions.return_value = left_pos  # type: ignore[attr-defined]
        robot._right._driver.get_all_positions.return_value = right_pos  # type: ignore[attr-defined]

        obs = robot.get_observation()
        np.testing.assert_array_almost_equal(obs.joint_positions[:7], left_pos)
        np.testing.assert_array_almost_equal(obs.joint_positions[7:], right_pos)

    def test_observation_merges_sensor_data_by_key_not_order(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")

        left_obs = MagicMock()
        right_obs = MagicMock()
        left_obs.joint_positions = np.zeros(7, dtype=np.float32)
        right_obs.joint_positions = np.ones(7, dtype=np.float32)
        left_obs.timestamp = 1.0
        right_obs.timestamp = 2.0
        left_obs.sensor_data = {
            "velocities": np.arange(7, dtype=np.float32),
            "efforts": np.arange(10, 17, dtype=np.float32),
        }
        right_obs.sensor_data = {
            "efforts": np.arange(20, 27, dtype=np.float32),
            "velocities": np.arange(30, 37, dtype=np.float32),
        }

        robot._left.get_observation = MagicMock(return_value=left_obs)  # type: ignore[attr-defined]
        robot._right.get_observation = MagicMock(return_value=right_obs)  # type: ignore[attr-defined]

        obs = robot.get_observation()
        assert obs.sensor_data is not None
        np.testing.assert_array_equal(
            obs.sensor_data["velocities"],
            np.concatenate([left_obs.sensor_data["velocities"], right_obs.sensor_data["velocities"]]),
        )
        np.testing.assert_array_equal(
            obs.sensor_data["efforts"],
            np.concatenate([left_obs.sensor_data["efforts"], right_obs.sensor_data["efforts"]]),
        )


# ---------------------------------------------------------------------------
# send_action
# ---------------------------------------------------------------------------


class TestBimanualWidowXAISendAction:
    def test_follower_splits_action(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")

        action = np.zeros(14, dtype=np.float32)
        # Should not raise
        robot.send_action(action)

        robot._left._driver.set_all_positions.assert_called()  # type: ignore[attr-defined]
        robot._right._driver.set_all_positions.assert_called()  # type: ignore[attr-defined]

    def test_wrong_shape_raises(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")
        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(7, dtype=np.float32))

    def test_leader_raises(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="leader")
        with pytest.raises(RuntimeError, match="Cannot send actions to a leader robot"):
            robot.send_action(np.zeros(14, dtype=np.float32))


# ---------------------------------------------------------------------------
# set_external_efforts
# ---------------------------------------------------------------------------


class TestBimanualWidowXAISetEfforts:
    def test_leader_splits_efforts(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="leader")

        efforts = np.ones(14, dtype=np.float32)
        robot.set_external_efforts(efforts, gain=1.0)

        robot._left._driver.set_all_external_efforts.assert_called()  # type: ignore[attr-defined]
        robot._right._driver.set_all_external_efforts.assert_called()  # type: ignore[attr-defined]

    def test_wrong_shape_raises(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="leader")
        with pytest.raises(ValueError, match="Expected efforts shape"):
            robot.set_external_efforts(np.zeros(7, dtype=np.float32))

    def test_follower_raises(self, mock_trossen_arm: MagicMock) -> None:
        robot = _make_bimanual(mock_trossen_arm, role="follower")
        with pytest.raises(RuntimeError, match="only available for leader robots"):
            robot.set_external_efforts(np.zeros(14, dtype=np.float32))
