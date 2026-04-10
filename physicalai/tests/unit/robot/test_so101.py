# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SO-101 robot driver.

All hardware communication is mocked — these tests verify the driver's logic
without requiring a physical robot or the feetech-servo-sdk package.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures: mock the feetech SDK
# ---------------------------------------------------------------------------

SAMPLE_CALIBRATION: dict[str, Any] = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
    "shoulder_lift": {"id": 2, "drive_mode": 1, "homing_offset": 1024, "range_min": 669, "range_max": 3292},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": 2048, "range_min": 846, "range_max": 3069},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 2048, "range_min": 956, "range_max": 3311},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 2048, "range_min": 59, "range_max": 3946},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 2048, "range_min": 2026, "range_max": 3074},
}


def _make_mock_sdk() -> MagicMock:
    """Build a mock ``scservo_sdk`` module with minimal behaviour."""
    sdk = MagicMock()

    # PortHandler
    port_handler = MagicMock()
    port_handler.openPort.return_value = True
    port_handler.setBaudRate.return_value = True
    sdk.PortHandler.return_value = port_handler

    # PacketHandler
    packet_handler = MagicMock()
    packet_handler.ping.return_value = (0, 0, 0)  # model, comm_result, error
    packet_handler.write1ByteTxRx.return_value = (0, 0)
    sdk.PacketHandler.return_value = packet_handler

    # GroupSyncRead
    sync_read = MagicMock()
    sync_read.addParam.return_value = True
    sync_read.txRxPacket.return_value = 0
    sync_read.isAvailable.return_value = True
    # Return 2048 ticks for every servo by default
    sync_read.getData.return_value = 2048
    sdk.GroupSyncRead.return_value = sync_read

    # GroupSyncWrite
    sync_write = MagicMock()
    sync_write.addParam.return_value = True
    sync_write.txPacket.return_value = 0
    sdk.GroupSyncWrite.return_value = sync_write

    return sdk


@pytest.fixture
def mock_sdk() -> Generator[MagicMock, None, None]:
    """Provide a mock scservo_sdk patched onto the SO101 module."""
    sdk = _make_mock_sdk()
    with (
        patch("physicalai.robot.so101.so101.PortHandler", sdk.PortHandler),
        patch("physicalai.robot.so101.so101.PacketHandler", sdk.PacketHandler),
        patch("physicalai.robot.so101.so101.GroupSyncRead", sdk.GroupSyncRead),
        patch("physicalai.robot.so101.so101.GroupSyncWrite", sdk.GroupSyncWrite),
    ):
        yield sdk


@pytest.fixture
def calibration_file(tmp_path: Path) -> Path:
    """Write a sample calibration JSON file and return its path."""
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(SAMPLE_CALIBRATION), encoding="utf-8")
    return path


@pytest.fixture
def calibration_obj() -> Any:
    """Build a typed SO-101 calibration object from in-memory sample data."""
    from physicalai.robot.so101 import SO101Calibration

    return SO101Calibration.from_dict(SAMPLE_CALIBRATION)


def _create_robot(
    mock_sdk: MagicMock,
    role: Literal["leader", "follower"] = "follower",
    calibration: Any | None = None,
) -> Any:
    """Instantiate SO101 with the mocked SDK."""
    from physicalai.robot.so101 import SO101, SO101Calibration

    if calibration is None:
        calibration = SO101Calibration.from_dict(SAMPLE_CALIBRATION)

    return SO101(
        port="/dev/ttyUSB0",
        role=role,
        calibration=calibration,
    )


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestSO101Construction:
    """Tests for SO101.__init__."""

    def test_invalid_role_raises(self, mock_sdk: MagicMock) -> None:
        """An invalid role string raises ValueError at construction."""
        with pytest.raises(ValueError, match="Invalid role"):
            _create_robot(mock_sdk, role="invalid")  # pyrefly: ignore[bad-argument-type]

    def test_servo_ids_derived_from_calibration(self, mock_sdk: MagicMock) -> None:
        """Servo IDs are derived from calibration in calibrated mode."""
        robot = _create_robot(mock_sdk)
        assert robot.servo_ids == {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }

    def test_calibration_rejects_non_positive_servo_id(self, mock_sdk: MagicMock) -> None:
        """Calibration with a non-positive servo ID raises at parse time."""
        bad_calibration = json.loads(json.dumps(SAMPLE_CALIBRATION))
        bad_calibration["gripper"]["id"] = 0

        from physicalai.robot.so101 import SO101Calibration

        with pytest.raises(ValueError, match="positive integers"):
            SO101Calibration.from_dict(bad_calibration)

    def test_calibration_rejects_duplicate_servo_ids(self, mock_sdk: MagicMock) -> None:
        """Calibration with duplicate servo IDs raises at parse time."""
        bad_calibration = json.loads(json.dumps(SAMPLE_CALIBRATION))
        bad_calibration["gripper"]["id"] = bad_calibration["wrist_roll"]["id"]

        from physicalai.robot.so101 import SO101Calibration

        with pytest.raises(ValueError, match="unique"):
            SO101Calibration.from_dict(bad_calibration)

    def test_none_calibration_raises(self, mock_sdk: MagicMock) -> None:
        """Main constructor rejects missing calibration."""
        from physicalai.robot.so101 import SO101

        with pytest.raises(ValueError, match="calibration is required"):
            SO101(port="/dev/ttyUSB0", calibration=None)

    def test_uncalibrated_factory_sets_ticks_mode(self, mock_sdk: MagicMock) -> None:
        """uncalibrated() creates an explicit raw-ticks mode robot."""
        from physicalai.robot.so101 import SO101

        robot = SO101.uncalibrated(port="/dev/ttyUSB0")

        assert robot.calibrated is False
        assert robot.unit == "ticks"
        assert robot.servo_ids == {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }


# ---------------------------------------------------------------------------
# Tests: connect / disconnect lifecycle
# ---------------------------------------------------------------------------


class TestSO101Lifecycle:
    """Tests for connect() and disconnect()."""

    def test_connect_opens_port_and_pings(self, mock_sdk: MagicMock) -> None:
        """connect() opens port, sets baudrate, pings servos, configures torque."""
        robot = _create_robot(mock_sdk)
        robot.connect()

        mock_sdk.PortHandler.return_value.openPort.assert_called_once()
        mock_sdk.PortHandler.return_value.setBaudRate.assert_called_once_with(1_000_000)
        # 6 servos pinged
        assert mock_sdk.PacketHandler.return_value.ping.call_count == 6

    def test_connect_is_idempotent(self, mock_sdk: MagicMock) -> None:
        """Calling connect() twice does not re-open the port."""
        robot = _create_robot(mock_sdk)
        robot.connect()
        robot.connect()

        mock_sdk.PortHandler.return_value.openPort.assert_called_once()

    def test_disconnect_closes_port(self, mock_sdk: MagicMock) -> None:
        """disconnect() closes the serial port."""
        robot = _create_robot(mock_sdk)
        robot.connect()
        robot.disconnect()

        mock_sdk.PortHandler.return_value.closePort.assert_called_once()

    def test_disconnect_is_idempotent(self, mock_sdk: MagicMock) -> None:
        """Calling disconnect() when not connected is a no-op."""
        robot = _create_robot(mock_sdk)
        robot.disconnect()  # should not raise

    def test_follower_enables_torque(self, mock_sdk: MagicMock) -> None:
        """Follower role enables torque on connect."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        # Check that write1ByteTxRx was called with torque_enable=1
        calls = mock_sdk.PacketHandler.return_value.write1ByteTxRx.call_args_list
        torque_values = [c.args[3] if len(c.args) > 3 else c[0][3] for c in calls]
        assert all(v == 1 for v in torque_values)

    def test_leader_disables_torque(self, mock_sdk: MagicMock) -> None:
        """Leader role disables torque on connect."""
        robot = _create_robot(mock_sdk, role="leader")
        robot.connect()

        calls = mock_sdk.PacketHandler.return_value.write1ByteTxRx.call_args_list
        torque_values = [c.args[3] if len(c.args) > 3 else c[0][3] for c in calls]
        assert all(v == 0 for v in torque_values)

    def test_port_open_failure_raises(self, mock_sdk: MagicMock) -> None:
        """ConnectionError if port cannot be opened."""
        mock_sdk.PortHandler.return_value.openPort.return_value = False
        robot = _create_robot(mock_sdk)

        with pytest.raises(ConnectionError, match="Failed to open"):
            robot.connect()

    def test_servo_ping_failure_raises(self, mock_sdk: MagicMock) -> None:
        """ConnectionError if a servo doesn't respond."""
        mock_sdk.PacketHandler.return_value.ping.return_value = (0, 1, 0)
        robot = _create_robot(mock_sdk)

        with pytest.raises(ConnectionError, match="did not respond"):
            robot.connect()


# ---------------------------------------------------------------------------
# Tests: observation / action
# ---------------------------------------------------------------------------


class TestSO101Observation:
    """Tests for get_observation()."""

    def test_observation_structure_uncalibrated(self, mock_sdk: MagicMock) -> None:
        """Uncalibrated observation has correct structure and dtype."""
        from physicalai.robot.so101 import SO101

        robot = SO101.uncalibrated(port="/dev/ttyUSB0")
        robot.connect()
        obs = robot.get_observation()

        assert isinstance(obs.joint_positions, np.ndarray)
        assert obs.joint_positions.shape == (6,)
        assert obs.joint_positions.dtype == np.float32
        assert isinstance(obs.timestamp, float)

    def test_observation_calibrated(self, mock_sdk: MagicMock, calibration_obj: Any) -> None:
        """Calibrated observation returns radians."""
        robot = _create_robot(mock_sdk, calibration=calibration_obj)
        robot.connect()
        obs = robot.get_observation()

        # All servos return 2048 ticks. For joints with homing_offset=2048,
        # the result should be 0.0 radians.
        state = obs.joint_positions
        assert state.shape == (6,)
        assert state.dtype == np.float32
        # shoulder_pan: (2048 - 2048) * 1 * radians_per_tick = 0.0
        assert state[0] == pytest.approx(0.0, abs=1e-6)
        # shoulder_lift: (2048 - 1024) * -1 * radians_per_tick ≈ -1.5708
        assert state[1] == pytest.approx(-1024 * 2 * np.pi / 4096, abs=1e-4)


class TestSO101Action:
    """Tests for send_action()."""

    def test_send_action_follower(self, mock_sdk: MagicMock) -> None:
        """Follower can send actions."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()
        action = np.zeros(6, dtype=np.float32)
        robot.send_action(action)

        mock_sdk.GroupSyncWrite.return_value.txPacket.assert_called_once()

    def test_send_action_leader_raises(self, mock_sdk: MagicMock) -> None:
        """Leader raises RuntimeError on send_action()."""
        robot = _create_robot(mock_sdk, role="leader")
        robot.connect()

        with pytest.raises(RuntimeError, match="Cannot send actions to a leader arm"):
            robot.send_action(np.zeros(6, dtype=np.float32))

    def test_send_action_wrong_shape_raises(self, mock_sdk: MagicMock) -> None:
        """ValueError on wrong action shape."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        with pytest.raises(ValueError, match="Expected action shape"):
            robot.send_action(np.zeros(3, dtype=np.float32))

    def test_send_action_calibrated_clamps(self, mock_sdk: MagicMock, calibration_obj: Any) -> None:
        """Calibrated send_action clamps to joint range limits."""
        robot = _create_robot(mock_sdk, role="follower", calibration=calibration_obj)
        robot.connect()
        # gripper range_max is 3074 ticks; sending 10.0 rad should be clamped internally
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
        robot.send_action(action)

        # Just verify it didn't crash — the clamping is internal
        mock_sdk.GroupSyncWrite.return_value.txPacket.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: calibration
# ---------------------------------------------------------------------------


class TestSO101Calibration:
    """Tests for calibration loading and conversion."""

    def test_constructor_with_calibration_path_valid(self, mock_sdk: MagicMock, calibration_file: Path) -> None:
        """Valid calibration path passed to constructor loads typed calibration."""
        from physicalai.robot.so101 import SO101

        robot = SO101(port="/dev/ttyUSB0", calibration=calibration_file)

        assert robot._calibration is not None  # noqa: SLF001
        assert "shoulder_pan" in robot._calibration.joints  # noqa: SLF001

    def test_from_path_missing_joint(self, tmp_path: Path) -> None:
        """Calibration file missing a joint raises ValueError."""
        bad_cal = {
            "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 2048, "range_min": 707, "range_max": 3439},
            # Missing 5 joints
        }
        path = tmp_path / "bad_cal.json"
        path.write_text(json.dumps(bad_cal), encoding="utf-8")

        from physicalai.robot.so101 import SO101Calibration

        with pytest.raises(ValueError, match="missing joints"):
            SO101Calibration.from_path(path)

    def test_from_path_not_a_dict(self, tmp_path: Path) -> None:
        """Calibration file that is a JSON array (not a dict) raises TypeError."""
        path = tmp_path / "bad_format.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        from physicalai.robot.so101 import SO101Calibration

        with pytest.raises(TypeError, match="JSON object"):
            SO101Calibration.from_path(path)

    def test_from_path_bad_drive_mode(self, tmp_path: Path) -> None:
        """Calibration with drive_mode not in {0, 1} raises ValueError."""
        bad_cal = {
            name: {**v, "drive_mode": 2} if name == "shoulder_pan" else v
            for name, v in SAMPLE_CALIBRATION.items()
        }
        path = tmp_path / "bad_drive_mode.json"
        path.write_text(json.dumps(bad_cal), encoding="utf-8")

        from physicalai.robot.so101 import SO101Calibration

        with pytest.raises(ValueError, match="drive_mode must be 0 or 1"):
            SO101Calibration.from_path(path)

    def test_tick_radian_roundtrip(self, mock_sdk: MagicMock, calibration_obj: Any) -> None:
        """Converting ticks → radians → ticks should roundtrip."""
        robot = _create_robot(mock_sdk, calibration=calibration_obj)

        original_ticks = np.array([2048, 1524, 2048, 2048, 2048, 2200], dtype=np.int32)
        radians = robot._ticks_to_radians(original_ticks)  # noqa: SLF001
        recovered_ticks = robot._radians_to_ticks(radians)  # noqa: SLF001
        np.testing.assert_array_almost_equal(original_ticks, recovered_ticks, decimal=0)
