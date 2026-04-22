# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for BimanualWidowXAIAdapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from physicalai.robot.trossen.constants import WIDOWXAI_JOINT_ORDER

from robots.widowxai.adapter import WidowXAIAdapter
from robots.widowxai.bimanual_adapter import BimanualWidowXAIAdapter
from schemas.robot import RobotType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inner_adapter(mode="follower"):
    """Return a fully-mocked WidowXAIAdapter."""
    adapter = MagicMock(spec=WidowXAIAdapter)
    adapter.is_controlled = mode == "follower"
    type(adapter).is_connected = PropertyMock(return_value=True)
    adapter.features.return_value = [f"{n}.pos" for n in WIDOWXAI_JOINT_ORDER] + [
        f"{n}.vel" for n in WIDOWXAI_JOINT_ORDER
    ]

    # Async methods
    adapter.connect = AsyncMock()
    adapter.disconnect = AsyncMock()
    adapter.enable_torque = AsyncMock(return_value={"event": "torque_was_enabled"})
    adapter.disable_torque = AsyncMock(return_value={"event": "torque_was_disabled"})
    adapter.set_joints_state = AsyncMock(return_value={"event": "joints_state_was_set", "joints": {}})
    adapter.set_forces = AsyncMock(return_value={})

    # read_state returns a realistic event dict
    joint_names = list(WIDOWXAI_JOINT_ORDER)
    state = {f"{n}.pos": 0.0 for n in joint_names}
    state.update({f"{n}.vel": 0.0 for n in joint_names})
    adapter.read_state = AsyncMock(
        return_value={
            "event": "state_was_updated",
            "state": state,
            "is_controlled": mode == "follower",
            "timestamp": 1000.0,
        }
    )

    forces = {f"{n}.eff": 0.0 for n in joint_names}
    adapter.read_forces = AsyncMock(
        return_value=(
            None
            if mode == "leader"
            else {
                "event": "force_was_updated",
                "state": forces,
                "is_controlled": True,
                "timestamp": 1000.0,
            }
        )
    )
    return adapter


def _make_bimanual(mode="follower"):
    left = _make_inner_adapter(mode)
    right = _make_inner_adapter(mode)
    bimanual = BimanualWidowXAIAdapter(left=left, right=right, mode=mode)
    return bimanual, left, right


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_robot_type_follower(self):
        bim, _, _ = _make_bimanual("follower")
        assert bim.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER

    def test_robot_type_leader(self):
        bim, _, _ = _make_bimanual("leader")
        assert bim.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER

    def test_is_connected_true_when_both_connected(self):
        bim, left, right = _make_bimanual()
        type(left).is_connected = PropertyMock(return_value=True)
        type(right).is_connected = PropertyMock(return_value=True)
        assert bim.is_connected is True

    def test_is_connected_false_when_one_disconnected(self):
        bim, left, right = _make_bimanual()
        type(left).is_connected = PropertyMock(return_value=False)
        type(right).is_connected = PropertyMock(return_value=True)
        assert bim.is_connected is False

    def test_features_prefixed(self):
        bim, _, _ = _make_bimanual()
        features = bim.features()
        assert all(f.startswith(("left_", "right_")) for f in features)
        # Total count = 2 arms x (7 pos + 7 vel) = 28
        assert len(features) == 28


# ---------------------------------------------------------------------------
# Connect / Disconnect
# ---------------------------------------------------------------------------


class TestConnectDisconnect:
    def test_connect_delegates_to_both(self):
        bim, left, right = _make_bimanual()
        asyncio.run(bim.connect())
        left.connect.assert_awaited_once()
        right.connect.assert_awaited_once()

    def test_disconnect_delegates_to_both(self):
        bim, left, right = _make_bimanual()
        asyncio.run(bim.disconnect())
        left.disconnect.assert_awaited_once()
        right.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------


class TestPing:
    def test_returns_pong(self):
        bim, _, _ = _make_bimanual()
        result = asyncio.run(bim.ping())
        assert result["event"] == "pong"


# ---------------------------------------------------------------------------
# read_state
# ---------------------------------------------------------------------------


class TestReadState:
    def test_keys_are_prefixed(self):
        bim, _, _ = _make_bimanual()
        result = asyncio.run(bim.read_state())
        assert result["event"] == "state_was_updated"
        for key in result["state"]:
            assert key.startswith(("left_", "right_")), key

    def test_both_arms_queried(self):
        bim, left, right = _make_bimanual()
        asyncio.run(bim.read_state())
        left.read_state.assert_awaited_once()
        right.read_state.assert_awaited_once()

    def test_merged_key_count(self):
        bim, _, _ = _make_bimanual()
        result = asyncio.run(bim.read_state())
        # 7 pos + 7 vel per arm x 2 arms = 28
        assert len(result["state"]) == 28


# ---------------------------------------------------------------------------
# read_forces
# ---------------------------------------------------------------------------


class TestReadForces:
    def test_follower_returns_prefixed_forces(self):
        bim, _, _ = _make_bimanual("follower")
        result = asyncio.run(bim.read_forces())
        assert result is not None
        for key in result["state"]:
            assert key.startswith(("left_", "right_")), key

    def test_leader_returns_none(self):
        bim, _, _ = _make_bimanual("leader")
        result = asyncio.run(bim.read_forces())
        assert result is None


# ---------------------------------------------------------------------------
# set_joints_state
# ---------------------------------------------------------------------------


class TestSetJointsState:
    def test_splits_and_delegates(self):
        bim, left, right = _make_bimanual()
        joints: dict[str, float] = {}
        for n in WIDOWXAI_JOINT_ORDER:
            joints[f"left_{n}.pos"] = 1.0
            joints[f"left_{n}.vel"] = 0.0
            joints[f"right_{n}.pos"] = 2.0
            joints[f"right_{n}.vel"] = 0.0

        asyncio.run(bim.set_joints_state(joints, goal_time=0.1))

        left_call_joints = left.set_joints_state.call_args[0][0]
        right_call_joints = right.set_joints_state.call_args[0][0]

        # Keys should be un-prefixed when handed to inner adapters
        assert all(not k.startswith("left_") and not k.startswith("right_") for k in left_call_joints)
        assert all(not k.startswith("left_") and not k.startswith("right_") for k in right_call_joints)

    def test_goal_time_forwarded(self):
        bim, left, right = _make_bimanual()
        joints = {f"left_{n}.pos": 0.0 for n in WIDOWXAI_JOINT_ORDER}
        joints.update({f"right_{n}.pos": 0.0 for n in WIDOWXAI_JOINT_ORDER})

        asyncio.run(bim.set_joints_state(joints, goal_time=0.25))

        assert left.set_joints_state.call_args[0][1] == pytest.approx(0.25)
        assert right.set_joints_state.call_args[0][1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# set_forces
# ---------------------------------------------------------------------------


class TestSetForces:
    def test_splits_and_delegates(self):
        bim, left, right = _make_bimanual("leader")
        forces = {f"left_{n}.eff": 0.1 for n in WIDOWXAI_JOINT_ORDER}
        forces.update({f"right_{n}.eff": 0.2 for n in WIDOWXAI_JOINT_ORDER})

        asyncio.run(bim.set_forces(forces))

        left_forces = left.set_forces.call_args[0][0]
        right_forces = right.set_forces.call_args[0][0]
        assert all(not k.startswith("left_") and not k.startswith("right_") for k in left_forces)
        assert all(not k.startswith("left_") and not k.startswith("right_") for k in right_forces)


# ---------------------------------------------------------------------------
# Torque
# ---------------------------------------------------------------------------


class TestTorque:
    def test_enable_torque(self):
        bim, left, right = _make_bimanual()
        result = asyncio.run(bim.enable_torque())
        assert result["event"] == "torque_was_enabled"
        left.enable_torque.assert_awaited_once()
        right.enable_torque.assert_awaited_once()

    def test_disable_torque(self):
        bim, left, right = _make_bimanual()
        result = asyncio.run(bim.disable_torque())
        assert result["event"] == "torque_was_disabled"
        left.disable_torque.assert_awaited_once()
        right.disable_torque.assert_awaited_once()
