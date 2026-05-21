# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for bimanual usage via PhysicalAIRobotAdapter."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from physicalai.robot.trossen.constants import WIDOWXAI_JOINT_ORDER

from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from schemas.robot import RobotType

NUM_JOINTS = len(WIDOWXAI_JOINT_ORDER)


def _make_bimanual_robot(role: str = "follower") -> MagicMock:
    joint_names = [f"left_{n}" for n in WIDOWXAI_JOINT_ORDER] + [f"right_{n}" for n in WIDOWXAI_JOINT_ORDER]
    robot = MagicMock()
    robot.joint_names = joint_names
    robot.is_connected.return_value = True
    robot.role = role

    velocities = np.zeros(2 * NUM_JOINTS, dtype=np.float32)
    efforts = np.zeros(2 * NUM_JOINTS, dtype=np.float32)
    positions = np.zeros(2 * NUM_JOINTS, dtype=np.float32)

    obs = MagicMock()
    obs.joint_positions = positions
    obs.timestamp = 1000.0
    obs.sensor_data = {"velocities": velocities, "efforts": efforts}
    robot.get_observation.return_value = obs
    return robot


def _make_adapter(mode: str = "follower") -> tuple[PhysicalAIRobotAdapter, MagicMock]:
    robot = _make_bimanual_robot(mode)
    robot_type = (
        RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER
        if mode == "follower"
        else RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER
    )
    adapter = PhysicalAIRobotAdapter(
        robot=robot,
        robot_type=robot_type,
        config=PhysicalAIRobotAdapterConfig(
            include_velocities=True,
            goal_time_scale=1.0,
            external_effort_gain=0.1,
        ),
    )
    return adapter, robot


class TestProperties:
    def test_robot_type_follower(self):
        adapter, _ = _make_adapter("follower")
        assert adapter.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER

    def test_robot_type_leader(self):
        adapter, _ = _make_adapter("leader")
        assert adapter.robot_type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER

    def test_features_prefixed(self):
        adapter, _ = _make_adapter("follower")
        features = adapter.features()
        assert all(f.startswith(("left_", "right_")) for f in features)
        assert len(features) == 28


class TestStateAndActions:
    def test_read_state_includes_prefixed_pos_and_vel(self):
        adapter, _ = _make_adapter("follower")
        result = asyncio.run(adapter.read_state())
        assert result["event"] == "state_was_updated"
        assert len(result["state"]) == 28
        for key in result["state"]:
            assert key.startswith(("left_", "right_"))

    def test_set_joints_state_calls_send_action(self):
        adapter, robot = _make_adapter("follower")
        joints: dict[str, float] = {}
        for n in WIDOWXAI_JOINT_ORDER:
            joints[f"left_{n}.pos"] = 1.0
            joints[f"right_{n}.pos"] = 2.0

        result = asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))
        assert result["event"] == "joints_state_was_set"
        robot.send_action.assert_called_once()
        call = robot.send_action.call_args
        assert call is not None
        assert call.args[0].shape == (14,)
        assert call.kwargs["goal_time"] == pytest.approx(0.1)


class TestForces:
    def test_read_forces_follower_returns_event(self):
        adapter, _ = _make_adapter("follower")
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        assert result["event"] == "force_was_updated"
        assert len(result["state"]) == 14

    def test_set_forces_leader_calls_set_external_efforts(self):
        adapter, robot = _make_adapter("leader")
        robot.set_external_efforts = MagicMock()
        forces = {f"left_{n}.eff": 0.1 for n in WIDOWXAI_JOINT_ORDER}
        forces.update({f"right_{n}.eff": 0.2 for n in WIDOWXAI_JOINT_ORDER})

        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces
        robot.set_external_efforts.assert_called_once()

    def test_set_forces_follower_noops(self):
        adapter, robot = _make_adapter("follower")
        robot.set_external_efforts = MagicMock()
        forces = {"left_gripper.eff": 1.0}
        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces
        robot.set_external_efforts.assert_not_called()
