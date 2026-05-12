# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from physicalai.robot.trossen.constants import WIDOWXAI_JOINT_ORDER
from physicalai.robot.trossen.widowxai import WidowXAI, WidowXAIObservation

from robots.widowxai.adapter import WidowXAIAdapter
from schemas.robot import RobotType


def _make_mock_robot(role="follower"):
    robot = MagicMock(spec=WidowXAI)
    robot.is_connected.return_value = False
    robot.joint_names = list(WIDOWXAI_JOINT_ORDER)
    type(robot).ip = PropertyMock(return_value="192.168.1.2")
    return robot


def _make_adapter(mode="follower"):
    robot = _make_mock_robot(role=mode)
    adapter = WidowXAIAdapter(robot=robot, mode=mode)
    return adapter, robot


def _make_observation(positions=None, velocities=None, efforts=None):
    obs = MagicMock(spec=WidowXAIObservation)
    obs.joint_positions = np.array(positions if positions is not None else [0.0] * 7, dtype=np.float32)
    obs.timestamp = 1000.0
    sensor_data = {"velocities": np.array(velocities if velocities is not None else [0.0] * 7, dtype=np.float32)}
    if efforts is not None:
        sensor_data["efforts"] = np.array(efforts, dtype=np.float32)
    obs.sensor_data = sensor_data
    return obs


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "WidowXAI"

    def test_robot_type_follower(self):
        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER

    def test_robot_type_leader(self):
        adapter, _ = _make_adapter(mode="leader")
        assert adapter.robot_type == RobotType.TROSSEN_WIDOWXAI_LEADER

    def test_is_connected_delegates(self):
        adapter, robot = _make_adapter()
        robot.is_connected.return_value = True
        assert adapter.is_connected is True
        robot.is_connected.return_value = False
        assert adapter.is_connected is False

    def test_features_includes_pos_and_vel(self):
        adapter, _ = _make_adapter()
        expected = [f"{n}.pos" for n in WIDOWXAI_JOINT_ORDER] + [f"{n}.vel" for n in WIDOWXAI_JOINT_ORDER]
        assert adapter.features() == expected


class TestDegreeRadianConversion:
    def test_read_state_converts_to_degrees_for_non_gripper(self):
        adapter, robot = _make_adapter()
        radian_values = [1.0, 0.5, -0.5, 1.5, -1.0, 0.3, 0.02]
        obs = _make_observation(positions=radian_values)
        robot.get_observation.return_value = obs

        result = asyncio.run(adapter.read_state())

        for i, name in enumerate(WIDOWXAI_JOINT_ORDER):
            if name == "gripper":
                assert result["state"]["gripper.pos"] == pytest.approx(0.02, abs=0.001)
            else:
                assert result["state"][f"{name}.pos"] == pytest.approx(np.rad2deg(radian_values[i]), abs=0.01)

    def test_set_joints_state_converts_degrees_to_radians(self):
        adapter, robot = _make_adapter()

        joints = {
            "shoulder_pan.pos": 57.2958,
            "shoulder_lift.pos": 28.6479,
            "elbow_flex.pos": -28.6479,
            "wrist_flex.pos": 85.9437,
            "wrist_yaw.pos": -57.2958,
            "wrist_roll.pos": 17.1887,
            "gripper.pos": 0.05,
            "shoulder_pan.vel": 0.0,
            "shoulder_lift.vel": 0.0,
            "elbow_flex.vel": 0.0,
            "wrist_flex.vel": 0.0,
            "wrist_yaw.vel": 0.0,
            "wrist_roll.vel": 0.0,
            "gripper.vel": 0.0,
        }

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        positions_sent = robot.send_action.call_args[0][0]

        expected_radians = [1.0, 0.5, -0.5, 1.5, -1.0, 0.3]
        for i, expected_rad in enumerate(expected_radians):
            assert positions_sent[i] == pytest.approx(expected_rad, abs=0.01)

        # Gripper is unchanged (raw value)
        assert positions_sent[6] == pytest.approx(0.05, abs=0.001)

    def test_roundtrip_conversion(self):
        adapter, robot = _make_adapter()
        original_radians = [0.5, -0.3, 1.0, -1.0, 0.2, 0.8, 0.05]

        obs = _make_observation(positions=original_radians)
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())

        joints = {}
        for name in WIDOWXAI_JOINT_ORDER:
            joints[f"{name}.pos"] = result["state"][f"{name}.pos"]
            joints[f"{name}.vel"] = 0.0

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        positions_sent = robot.send_action.call_args[0][0]

        for i, expected_rad in enumerate(original_radians):
            assert positions_sent[i] == pytest.approx(expected_rad, abs=0.01)


class TestConnect:
    def test_connect_calls_driver(self):
        adapter, robot = _make_adapter()
        asyncio.run(adapter.connect())
        robot.connect.assert_called_once()

    def test_connect_follower_sets_controlled(self):
        adapter, robot = _make_adapter(mode="follower")
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is True

    def test_connect_leader_unsets_controlled(self):
        adapter, robot = _make_adapter(mode="leader")
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is False


class TestDisconnect:
    def test_disconnect_calls_driver(self):
        adapter, robot = _make_adapter()
        asyncio.run(adapter.disconnect())
        robot.disconnect.assert_called_once()


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestSetJointsState:
    def test_calls_send_action_with_positions_and_goal_time(self):
        adapter, robot = _make_adapter()
        joints = {}
        for name in WIDOWXAI_JOINT_ORDER:
            joints[f"{name}.pos"] = 0.0
            joints[f"{name}.vel"] = 0.1

        asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))

        robot.send_action.assert_called_once()
        call_args = robot.send_action.call_args
        assert call_args[0][0].shape == (7,)
        # Goal time is set higher to prevent oscillations
        assert call_args[1]["goal_time"] == 3 * 0.1

    def test_raises_for_leader(self):
        adapter, _ = _make_adapter(mode="leader")
        joints = {f"{name}.pos": 0.0 for name in WIDOWXAI_JOINT_ORDER}
        joints.update({f"{name}.vel": 0.0 for name in WIDOWXAI_JOINT_ORDER})
        with pytest.raises(RuntimeError):
            asyncio.run(adapter.set_joints_state(joints, goal_time=0.1))


class TestReadState:
    def test_returns_state_event(self):
        adapter, robot = _make_adapter()
        obs = _make_observation()
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())
        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "is_controlled" in result

    def test_state_has_pos_and_vel_keys(self):
        adapter, robot = _make_adapter()
        obs = _make_observation()
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_state())
        state = result["state"]
        for name in WIDOWXAI_JOINT_ORDER:
            assert f"{name}.pos" in state
            assert f"{name}.vel" in state


class TestReadForces:
    def test_follower_returns_forces(self):
        adapter, robot = _make_adapter(mode="follower")
        efforts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        obs = _make_observation(efforts=efforts)
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        state = result["state"]
        for name in WIDOWXAI_JOINT_ORDER:
            assert f"{name}.eff" in state

    def test_leader_returns_none(self):
        adapter, _ = _make_adapter(mode="leader")
        result = asyncio.run(adapter.read_forces())
        assert result is None


class TestSetForces:
    def test_leader_calls_set_external_efforts(self):
        adapter, robot = _make_adapter(mode="leader")
        forces = {f"{name}.eff": float(i) * 0.1 for i, name in enumerate(WIDOWXAI_JOINT_ORDER)}
        asyncio.run(adapter.set_forces(forces))

        robot.set_external_efforts.assert_called_once()
        call_args = robot.set_external_efforts.call_args
        efforts_sent = call_args[0][0]
        gain = call_args[0][1]

        assert gain == pytest.approx(0.1, abs=1e-6)
        for i, name in enumerate(WIDOWXAI_JOINT_ORDER):
            assert efforts_sent[i] == pytest.approx(forces[f"{name}.eff"], abs=1e-6)

    def test_follower_returns_forces_unchanged(self):
        adapter, _ = _make_adapter(mode="follower")
        forces = {f"{name}.eff": 0.5 for name in WIDOWXAI_JOINT_ORDER}
        result = asyncio.run(adapter.set_forces(forces))
        assert result == forces


class TestTorque:
    def test_enable_torque_event(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.enable_torque())
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque_event(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.disable_torque())
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False
