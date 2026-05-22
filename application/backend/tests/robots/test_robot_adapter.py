import asyncio
from unittest.mock import MagicMock

from physicalai.robot.so101.constants import SO101_JOINT_ORDER

from robots.physicalai_adapter import PhysicalAIRobotAdapter, PhysicalAIRobotAdapterConfig
from schemas.robot import RobotType


def _make_mock_robot() -> MagicMock:
    robot = MagicMock()
    robot.port = "/dev/ttyUSB0"
    robot.joint_names = list(SO101_JOINT_ORDER)
    obs = MagicMock()
    obs.joint_positions = [0.0] * len(SO101_JOINT_ORDER)
    obs.sensor_data = None
    robot.get_observation.return_value = obs
    robot.is_connected.return_value = False
    return robot


def _make_adapter(
    mode: str = "follower",
) -> tuple[PhysicalAIRobotAdapter, MagicMock]:
    robot = _make_mock_robot()
    robot_type = RobotType.SO101_FOLLOWER if mode == "follower" else RobotType.SO101_LEADER
    adapter = PhysicalAIRobotAdapter(
        robot=robot,
        robot_type=robot_type,
        config=PhysicalAIRobotAdapterConfig(
            include_velocities=False,
            goal_time_scale=1.0,
            external_effort_gain=None,
        ),
    )
    return adapter, robot


class TestProperties:
    def test_name(self):
        adapter, _ = _make_adapter()
        assert adapter.name == "PhysicalAIRobot"

    def test_robot_type_follower(self):
        adapter, _ = _make_adapter(mode="follower")
        assert adapter.robot_type == RobotType.SO101_FOLLOWER

    def test_robot_type_teleoperator(self):
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
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(SO101_JOINT_ORDER)
        obs.sensor_data = None
        robot.get_observation.return_value = obs

        result = asyncio.run(adapter.read_state())

        assert result["event"] == "state_was_updated"
        assert "state" in result
        assert "timestamp" in result
        state = result["state"]
        assert len(state) == 6
        for name in SO101_JOINT_ORDER:
            assert f"{name}.pos" in state
        robot.get_observation.assert_called_once()


class TestSetJointsState:
    def test_sends_action_to_robot(self):
        adapter, robot = _make_adapter()

        joints = {f"{name}.pos": 0.0 for name in SO101_JOINT_ORDER}
        result = asyncio.run(adapter.set_joints_state(joints, goal_time=0.033))

        assert result["event"] == "joints_state_was_set"
        robot.send_action.assert_called_once()

    def test_delegates_state_send_to_driver(self):
        adapter, robot = _make_adapter()

        far_joints = {f"{name}.pos": 1000.0 for name in SO101_JOINT_ORDER}
        goal_time = 0.033
        asyncio.run(adapter.set_joints_state(far_joints, goal_time=goal_time))
        robot.send_action.assert_called_once()


class TestTorque:
    def test_enable_torque(self):
        adapter, robot = _make_adapter()
        robot.set_torque = MagicMock()
        result = asyncio.run(adapter.enable_torque())
        robot.set_torque.assert_called_once_with(enabled=True)
        assert result["event"] == "torque_was_enabled"
        assert adapter.is_controlled is True

    def test_disable_torque(self):
        adapter, robot = _make_adapter()
        robot.set_torque = MagicMock()
        result = asyncio.run(adapter.disable_torque())
        robot.set_torque.assert_called_once_with(enabled=False)
        assert result["event"] == "torque_was_disabled"
        assert adapter.is_controlled is False

    def test_torque_noops_when_driver_has_no_set_torque(self):
        adapter, robot = _make_adapter()
        if hasattr(robot, "set_torque"):
            delattr(robot, "set_torque")

        enable = asyncio.run(adapter.enable_torque())
        disable = asyncio.run(adapter.disable_torque())

        assert enable["event"] == "torque_was_enabled"
        assert disable["event"] == "torque_was_disabled"


class TestPing:
    def test_ping_returns_pong(self):
        adapter, _ = _make_adapter()
        result = asyncio.run(adapter.ping())
        assert result["event"] == "pong"
        assert "timestamp" in result


class TestReadForces:
    def test_returns_force_event_with_none_state(self):
        adapter, robot = _make_adapter()
        obs = MagicMock()
        obs.joint_positions = [0.0] * len(SO101_JOINT_ORDER)
        obs.sensor_data = None
        robot.get_observation.return_value = obs
        result = asyncio.run(adapter.read_forces())
        assert result is not None
        assert result["event"] == "force_was_updated"
        assert result["state"] is None


class TestSetForces:
    def test_noops_if_force_method_missing(self):
        adapter, robot = _make_adapter()
        forces = {"shoulder_pan.eff": 1.0}
        result = asyncio.run(adapter.set_forces(forces))

        assert result == forces
