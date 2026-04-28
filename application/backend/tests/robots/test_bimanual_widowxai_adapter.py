# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for BimanualWidowXAIAdapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from physicalai.robot.trossen.constants import WIDOWXAI_JOINT_ORDER

from robots.widowxai.bimanual_adapter import BimanualWidowXAIAdapter
from schemas.robot import RobotType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_JOINTS = len(WIDOWXAI_JOINT_ORDER)  # 7 per arm, 14 total


def _make_bimanual_robot(mode: str = "follower") -> MagicMock:
    """Return a mocked BimanualWidowXAI robot."""
    joint_names = [f"left_{n}" for n in WIDOWXAI_JOINT_ORDER] + [f"right_{n}" for n in WIDOWXAI_JOINT_ORDER]
    robot = MagicMock()
    robot.joint_names = joint_names
    robot.is_connected.return_value = True
    robot.role = mode

    velocities = np.zeros(2 * NUM_JOINTS, dtype=np.float32)
    efforts = np.zeros(2 * NUM_JOINTS, dtype=np.float32)
    positions = np.zeros(2 * NUM_JOINTS, dtype=np.float32)

    obs = MagicMock()
    obs.joint_positions = positions
    obs.timestamp = 1000.0
    obs.sensor_data = {"velocities": velocities, "efforts": efforts}
    robot.get_observation.return_value = obs

    return robot


def _make_adapter(mode: str = "follower") -> BimanualWidowXAIAdapter:
    robot = _make_bimanual_robot(mode)
    return BimanualWidowXAIAdapter(robot=robot, mode=mode)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_robot_type_follower(self):
        adapter = _make_adapter("follower")
        assert adapter.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER

    def test_robot_type_leader(self):
        adapter = _make_adapter("leader")
        assert adapter.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER

    def test_is_connected(self):
        adapter = _make_adapter()
        assert adapter.is_connected is True

    def test_is_connected_false_when_robot_not_connected(self):
        robot = _make_bimanual_robot()
        robot.is_connected.return_value = False
        adapter = BimanualWidowXAIAdapter(robot=robot, mode="follower")
        assert adapter.is_connected is False

    def test_features_prefixed(self):
        adapter = _make_adapter()
        features = adapter.features()
        assert all(f.startswith(("left_", "right_")) for f in features)
        # 2 arms x 7 joints x (pos + vel) = 28
        assert len(features) == 28


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    def test_connect_calls_robot(self):
        robot = _make_bimanual_robot()
        adapter = BimanualWidowXAIAdapter(robot=robot, mode="follower")
        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", new=AsyncMock()):
            asyncio.run(adapter.connect())
        assert adapter.is_controlled is True

    def test_disconnect_calls_robot(self):
        robot = _make_bimanual_robot()
        adapter = BimanualWidowXAIAdapter(robot=robot, mode="follower")
        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", new=AsyncMock()):
            asyncio.run(adapter.disconnect())


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_returns_pong(self):
        adapter = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"


# ---------------------------------------------------------------------------
# read_state
# ---------------------------------------------------------------------------


class TestReadState:
    def test_keys_are_prefixed(self):
        adapter = _make_adapter()
        obs = adapter._robot.get_observation()
        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", new=AsyncMock(return_value=obs)):
            result = asyncio.run(adapter.read_state())
        assert result["event"] == "state_was_updated"
        for key in result["state"]:
            assert key.startswith(("left_", "right_")), key

    def test_merged_key_count(self):
        adapter = _make_adapter()
        obs = adapter._robot.get_observation()
        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", new=AsyncMock(return_value=obs)):
            result = asyncio.run(adapter.read_state())
        # 14 pos + 14 vel = 28
        assert len(result["state"]) == 28


# ---------------------------------------------------------------------------
# read_forces
# ---------------------------------------------------------------------------


class TestReadForces:
    def test_follower_returns_prefixed_forces(self):
        adapter = _make_adapter("follower")
        obs = adapter._robot.get_observation()
        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", new=AsyncMock(return_value=obs)):
            result = asyncio.run(adapter.read_forces())
        assert result is not None
        assert result["event"] == "force_was_updated"
        for key in result["state"]:
            assert key.startswith(("left_", "right_")), key

    def test_leader_returns_none(self):
        adapter = _make_adapter("leader")
        result = asyncio.run(adapter.read_forces())
        assert result is None


# ---------------------------------------------------------------------------
# set_joints_state
# ---------------------------------------------------------------------------


class TestSetJointsState:
    def test_builds_action_and_calls_robot(self):
        adapter = _make_adapter("follower")
        joints: dict[str, float] = {}
        for n in WIDOWXAI_JOINT_ORDER:
            joints[f"left_{n}.pos"] = 45.0
            joints[f"right_{n}.pos"] = 90.0

        captured: list = []

        async def fake_to_thread(fn, *args, **kwargs):
            captured.append((fn, args, kwargs))

        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", side_effect=fake_to_thread):
            asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        assert len(captured) == 1
        fn, args, kwargs = captured[0]
        assert fn == adapter._robot.send_action
        action = args[0]
        assert action.shape == (14,)
        assert kwargs["goal_time"] == pytest.approx(0.1)

    def test_leader_raises(self):
        adapter = _make_adapter("leader")
        with pytest.raises(RuntimeError, match="Cannot send actions to a leader robot"):
            asyncio.run(adapter.set_joints_state({}, goal_time=0.1))


# ---------------------------------------------------------------------------
# set_forces
# ---------------------------------------------------------------------------


class TestSetForces:
    def test_leader_calls_robot(self):
        adapter = _make_adapter("leader")
        forces: dict[str, float] = {}
        for n in WIDOWXAI_JOINT_ORDER:
            forces[f"left_{n}.eff"] = 0.1
            forces[f"right_{n}.eff"] = 0.2

        captured: list = []

        async def fake_to_thread(fn, *args, **kwargs):
            captured.append((fn, args, kwargs))

        with patch("robots.widowxai.bimanual_adapter.asyncio.to_thread", side_effect=fake_to_thread):
            asyncio.run(adapter.set_forces(forces))

        assert len(captured) == 1
        fn, args, _ = captured[0]
        assert fn == adapter._robot.set_external_efforts
        efforts = args[0]
        assert efforts.shape == (14,)

    def test_follower_skips_and_returns_forces(self):
        adapter = _make_adapter("follower")
        forces = {"left_gripper.eff": 1.0}
        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces


# ---------------------------------------------------------------------------
# Torque
# ---------------------------------------------------------------------------


class TestTorque:
    def test_enable_torque(self):
        adapter = _make_adapter()
        result = asyncio.run(adapter.enable_torque())
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque(self):
        adapter = _make_adapter()
        result = asyncio.run(adapter.disable_torque())
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False
