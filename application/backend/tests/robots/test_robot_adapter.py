import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from physicalai.robot.so101.calibration import SO101Calibration, SO101JointCalibration
from physicalai.robot.so101.constants import RADIANS_PER_TICK, SO101_JOINT_ORDER

from robots.robot_adapter import SO101Adapter, _clamp, _clamp_joints
from schemas.calibration import Calibration, CalibrationValue


def _make_calibration_value(joint_id: int, drive_mode: int = 0, homing_offset: int = 2048) -> CalibrationValue:
    return CalibrationValue(
        id=joint_id,
        joint_name=SO101_JOINT_ORDER[joint_id - 1],
        drive_mode=drive_mode,
        homing_offset=homing_offset,
        range_min=0,
        range_max=4095,
    )


def _make_backend_calibration(drive_modes: dict[str, int] | None = None) -> Calibration:
    if drive_modes is None:
        drive_modes = {}

    values = {}
    for i, name in enumerate(SO101_JOINT_ORDER):
        dm = drive_modes.get(name, 0)
        values[name] = _make_calibration_value(i + 1, drive_mode=dm)

    cal = MagicMock(spec=Calibration)
    cal.values = values
    return cal


def _make_so101_calibration(drive_modes: dict[str, int] | None = None) -> SO101Calibration:
    if drive_modes is None:
        drive_modes = {}

    joints = {}
    for i, name in enumerate(SO101_JOINT_ORDER):
        dm = drive_modes.get(name, 0)
        joints[name] = SO101JointCalibration(
            id=i + 1,
            drive_mode=dm,
            homing_offset=2048,
            range_min=0,
            range_max=4095,
        )
    return SO101Calibration(joints=joints)


def _make_mock_robot(so101_cal: SO101Calibration | None = None) -> MagicMock:
    robot = MagicMock()
    robot.port = "/dev/ttyUSB0"
    robot._calibration = so101_cal or _make_so101_calibration()
    robot.is_connected.return_value = False
    return robot


def _make_adapter(
    mode: str = "follower",
    drive_modes: dict[str, int] | None = None,
) -> tuple[SO101Adapter, MagicMock]:
    backend_cal = _make_backend_calibration(drive_modes)
    so101_cal = _make_so101_calibration(drive_modes)
    robot = _make_mock_robot(so101_cal)
    adapter = SO101Adapter(robot=robot, mode=mode, calibration=backend_cal)
    return adapter, robot


class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0, 10.0) == 5.0

    def test_above_range(self):
        assert _clamp(15.0, 10.0) == 10.0

    def test_below_range(self):
        assert _clamp(-15.0, 10.0) == -10.0

    def test_zero(self):
        assert _clamp(0.0, 10.0) == 0.0


class TestClampJoints:
    def test_clamps_towards_target(self):
        current = {"a.pos": 0.0, "b.pos": 0.0}
        target = {"a.pos": 100.0, "b.pos": -100.0}
        result = _clamp_joints(current, target, 50.0)
        assert result["a.pos"] == 50.0
        assert result["b.pos"] == -50.0

    def test_no_clamp_when_within_range(self):
        current = {"a.pos": 90.0}
        target = {"a.pos": 100.0}
        result = _clamp_joints(current, target, 50.0)
        assert result["a.pos"] == 100.0


class TestNormalizationRoundtrip:
    def test_radians_to_normalized_and_back(self):
        adapter, _ = _make_adapter()

        radians_in = np.array([0.5, -0.5, 1.0, -1.0, 0.0, 0.3], dtype=np.float32)
        normalized = adapter._radians_to_normalized(radians_in)
        radians_out = adapter._normalized_to_radians(normalized)

        np.testing.assert_allclose(radians_in, radians_out, atol=1e-5)


class TestNormalizationValues:
    """Verify boundary values using radians that correspond to tick boundaries.

    With homing_offset=2048 and drive_mode=0:
        0 radians  → tick 2048 → normalized ~0 (body) / ~50 (gripper)
        tick 0     → radians = (0 - 2048) * 1 * RADIANS_PER_TICK
        tick 4095  → radians = (4095 - 2048) * 1 * RADIANS_PER_TICK
    """

    @staticmethod
    def _ticks_to_rad(tick: int, homing_offset: int = 2048) -> float:
        return (tick - homing_offset) * RADIANS_PER_TICK

    def test_zero_radians_gives_center_normalized_for_body(self):
        adapter, _ = _make_adapter()
        radians = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = adapter._radians_to_normalized(radians)

        for name in SO101_JOINT_ORDER:
            if name != "gripper":
                assert result[f"{name}.pos"] == pytest.approx(0.0, abs=0.05)

    def test_min_radians_gives_minus_100_for_body(self):
        adapter, _ = _make_adapter()
        rad = self._ticks_to_rad(0)
        radians = np.full(6, rad, dtype=np.float32)
        result = adapter._radians_to_normalized(radians)

        for name in SO101_JOINT_ORDER:
            if name != "gripper":
                assert result[f"{name}.pos"] == pytest.approx(-100.0, abs=0.1)

    def test_max_radians_gives_100_for_body(self):
        adapter, _ = _make_adapter()
        rad = self._ticks_to_rad(4095)
        radians = np.full(6, rad, dtype=np.float32)
        result = adapter._radians_to_normalized(radians)

        for name in SO101_JOINT_ORDER:
            if name != "gripper":
                assert result[f"{name}.pos"] == pytest.approx(100.0, abs=0.1)

    def test_min_radians_gives_zero_for_gripper(self):
        adapter, _ = _make_adapter()
        rad = self._ticks_to_rad(0)
        radians = np.full(6, rad, dtype=np.float32)
        result = adapter._radians_to_normalized(radians)
        assert result["gripper.pos"] == pytest.approx(0.0, abs=0.1)

    def test_max_radians_gives_100_for_gripper(self):
        adapter, _ = _make_adapter()
        rad_body = self._ticks_to_rad(0)
        rad_grip = self._ticks_to_rad(4095)
        radians = np.array([rad_body, rad_body, rad_body, rad_body, rad_body, rad_grip], dtype=np.float32)
        result = adapter._radians_to_normalized(radians)
        assert result["gripper.pos"] == pytest.approx(100.0, abs=0.1)


class TestDriveModeFlip:
    def test_drive_mode_1_flips_body_sign(self):
        adapter_normal, _ = _make_adapter(drive_modes={})
        adapter_flipped, _ = _make_adapter(drive_modes={"shoulder_pan": 1})

        # Same physical tick=3072 produces opposite radians under different drive_modes:
        #   drive_mode=0: rad = (3072-2048) * +1 * RPT
        #   drive_mode=1: rad = (3072-2048) * -1 * RPT
        rad_val = (3072 - 2048) * RADIANS_PER_TICK
        radians_normal = np.array([rad_val, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        radians_flipped = np.array([-rad_val, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        norm_normal = adapter_normal._radians_to_normalized(radians_normal)
        norm_flipped = adapter_flipped._radians_to_normalized(radians_flipped)

        assert norm_normal["shoulder_pan.pos"] == pytest.approx(-norm_flipped["shoulder_pan.pos"], abs=0.1)

    def test_drive_mode_1_flips_gripper(self):
        adapter_normal, _ = _make_adapter(drive_modes={})
        adapter_flipped, _ = _make_adapter(drive_modes={"gripper": 1})

        rad_val = (1024 - 2048) * RADIANS_PER_TICK
        radians_normal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, rad_val], dtype=np.float32)
        radians_flipped = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -rad_val], dtype=np.float32)

        norm_normal = adapter_normal._radians_to_normalized(radians_normal)
        norm_flipped = adapter_flipped._radians_to_normalized(radians_flipped)

        assert norm_normal["gripper.pos"] + norm_flipped["gripper.pos"] == pytest.approx(100.0, abs=0.1)


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "So101"

    def test_robot_type_follower(self):
        from schemas.robot import RobotType

        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.SO101_FOLLOWER

    def test_robot_type_teleoperator(self):
        from schemas.robot import RobotType

        adapter, _ = _make_adapter(mode="teleoperator")
        assert adapter.robot_type == RobotType.SO101_LEADER

    def test_is_connected_delegates_to_robot(self):
        adapter, robot = _make_adapter()
        robot.is_connected.return_value = True
        assert adapter.is_connected is True
        robot.is_connected.return_value = False
        assert adapter.is_connected is False

    def test_features(self):
        adapter, _ = _make_adapter()
        expected = [f"{name}.pos" for name in SO101_JOINT_ORDER]
        assert adapter.features() == expected


class TestConnect:
    def test_connect_calls_robot_connect(self):
        adapter, robot = _make_adapter()
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        robot.connect.assert_called_once()

    def test_connect_sets_is_controlled_for_follower(self):
        adapter, robot = _make_adapter(mode="follower")
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is True

    def test_connect_does_not_set_controlled_for_teleoperator(self):
        adapter, robot = _make_adapter(mode="teleoperator")
        robot.connect = MagicMock()
        asyncio.run(adapter.connect())
        assert adapter.is_controlled is False


class TestDisconnect:
    def test_disconnect_calls_robot_disconnect(self):
        adapter, robot = _make_adapter()
        robot.disconnect = MagicMock()
        asyncio.run(adapter.disconnect())
        robot.disconnect.assert_called_once()


class TestReadState:
    def test_returns_normalized_state_dict(self):
        adapter, robot = _make_adapter()
        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock

        result = asyncio.run(adapter.read_state())

        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "timestamp" in result
        state = result["state"]
        assert len(state) == 6
        for name in SO101_JOINT_ORDER:
            assert f"{name}.pos" in state


class TestSetJointsState:
    def test_sends_action_to_robot(self):
        adapter, robot = _make_adapter()

        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock
        robot.send_action = MagicMock()

        joints = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}
        result = asyncio.run(adapter.set_joints_state(joints, goal_time=0.033))

        assert result["event"] == "joints_state_was_set"
        robot.send_action.assert_called_once()
        action_array = robot.send_action.call_args[0][0]
        assert action_array.shape == (6,)

    def test_velocity_clamping_limits_movement(self):
        from physicalai.robot.so101.constants import MAX_SPEED_RAD_S

        adapter, robot = _make_adapter()

        obs_mock = MagicMock()
        obs_mock.joint_positions = np.zeros(6, dtype=np.float32)
        robot.get_observation.return_value = obs_mock
        robot.send_action = MagicMock()

        far_joints = {f"{name}.pos": 1000.0 for name in SO101_JOINT_ORDER}
        goal_time = 0.033
        asyncio.run(adapter.set_joints_state(far_joints, goal_time=goal_time))

        # Verify clamping happened in radian space
        max_rad = MAX_SPEED_RAD_S * goal_time
        action_sent = robot.send_action.call_args[0][0]
        for i in range(len(action_sent)):
            assert abs(action_sent[i]) <= max_rad + 1e-6


class TestTorque:
    def test_enable_torque(self):
        adapter, robot = _make_adapter()
        robot._set_torque = MagicMock()
        result = asyncio.run(adapter.enable_torque())
        robot._set_torque.assert_called_once_with(enabled=True)
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque(self):
        adapter, robot = _make_adapter()
        robot._set_torque = MagicMock()
        result = asyncio.run(adapter.disable_torque())
        robot._set_torque.assert_called_once_with(enabled=False)
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestReadForces:
    def test_returns_none_state(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.read_forces())
        assert result["event"] == "force_was_updated"
        assert result["state"] is None


class TestSetForces:
    def test_raises_not_implemented(self):
        adapter, _ = _make_adapter()
        with pytest.raises(NotImplementedError):
            asyncio.run(adapter.set_forces({}))
