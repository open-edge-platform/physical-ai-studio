# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SO-101 robot driver.

All hardware communication is mocked — these tests verify the driver's logic
without requiring a physical robot or the feetech-servo-sdk package.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.modules.setdefault("scservo_sdk", MagicMock())

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
    packet_handler.write2ByteTxRx.return_value = (0, 0)
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
    sys.modules.pop("physicalai.robot.so101.so101", None)
    sys.modules.pop("physicalai.robot.so101", None)
    with patch.dict(sys.modules, {"scservo_sdk": sdk}):
        import_module("physicalai.robot.so101.so101")
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
    from physicalai.robot.so101.calibration import SO101Calibration

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

        from physicalai.robot.so101.calibration import SO101Calibration

        with pytest.raises(ValueError, match="positive integers"):
            SO101Calibration.from_dict(bad_calibration)

    def test_calibration_rejects_duplicate_servo_ids(self, mock_sdk: MagicMock) -> None:
        """Calibration with duplicate servo IDs raises at parse time."""
        bad_calibration = json.loads(json.dumps(SAMPLE_CALIBRATION))
        bad_calibration["gripper"]["id"] = bad_calibration["wrist_roll"]["id"]

        from physicalai.robot.so101.calibration import SO101Calibration

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

    def test_default_unit_is_normalized(self, mock_sdk: MagicMock) -> None:
        """Calibrated SO101 defaults to normalized for observation/action units."""
        robot = _create_robot(mock_sdk)
        assert robot.unit == "normalized"

    def test_normalized_unit_allowed_in_calibrated_mode(self, mock_sdk: MagicMock) -> None:
        """calibrated mode accepts normalized unit."""
        from physicalai.robot.so101 import SO101

        robot = SO101(port="/dev/ttyUSB0", calibration=_create_robot(mock_sdk)._calibration, unit="normalized")
        assert robot.unit == "normalized"

    def test_uncalibrated_rejects_non_ticks_unit(self, mock_sdk: MagicMock) -> None:
        """uncalibrated mode only accepts ticks as unit."""
        from physicalai.robot.so101 import SO101

        with pytest.raises(ValueError, match="only supports unit='ticks'"):
            SO101.uncalibrated(port="/dev/ttyUSB0", unit="normalized")  # pyrefly: ignore[bad-argument-type]

    def test_calibrated_rejects_ticks_unit(self, mock_sdk: MagicMock) -> None:
        """calibrated mode does not accept ticks as unit."""
        from physicalai.robot.so101 import SO101, SO101Calibration

        calibration = SO101Calibration.from_dict(SAMPLE_CALIBRATION)
        with pytest.raises(ValueError, match="does not support unit='ticks'"):
            SO101(port="/dev/ttyUSB0", calibration=calibration, unit="ticks")


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
        """Follower role enables torque on connect (after servo configuration)."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        # Filter to torque writes only (address 40 = TORQUE_ENABLE)
        calls = mock_sdk.PacketHandler.return_value.write1ByteTxRx.call_args_list
        torque_calls = [c for c in calls if (c.args[2] if len(c.args) > 2 else c[0][2]) == 40]
        # Last 6 torque writes should be enable (value=1), preceding 6 are disable for config
        final_torque_values = [c.args[3] if len(c.args) > 3 else c[0][3] for c in torque_calls[-6:]]
        assert all(v == 1 for v in final_torque_values)

    def test_leader_disables_torque(self, mock_sdk: MagicMock) -> None:
        """Leader role disables torque on connect."""
        robot = _create_robot(mock_sdk, role="leader")
        robot.connect()

        # Filter to torque writes only (address 40 = TORQUE_ENABLE)
        calls = mock_sdk.PacketHandler.return_value.write1ByteTxRx.call_args_list
        torque_calls = [c for c in calls if (c.args[2] if len(c.args) > 2 else c[0][2]) == 40]
        # All torque writes should be disable (value=0) — config disables, leader keeps disabled
        torque_values = [c.args[3] if len(c.args) > 3 else c[0][3] for c in torque_calls]
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

    def test_set_torque_delegates_to_internal_impl(self, mock_sdk: MagicMock) -> None:
        """Public set_torque forwards to internal torque implementation."""
        robot = _create_robot(mock_sdk)
        robot.connect()
        robot._set_torque = MagicMock()  # noqa: SLF001

        robot.set_torque(enabled=True)
        robot._set_torque.assert_called_once_with(enabled=True)  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tests: servo configuration
# ---------------------------------------------------------------------------


class TestSO101ServoConfiguration:
    """Tests for _configure_servos() called during connect()."""

    def _get_write_calls(self, mock_sdk: MagicMock) -> tuple[list, list]:
        """Return (write1Byte_calls, write2Byte_calls) from packet_handler."""
        ph = mock_sdk.PacketHandler.return_value
        return ph.write1ByteTxRx.call_args_list, ph.write2ByteTxRx.call_args_list

    def _extract_register_writes(self, calls: list, address: int) -> list[tuple[int, int]]:
        """Extract (servo_id, value) pairs for writes to a specific address."""
        results = []
        for c in calls:
            args = c.args if c.args else c[0]
            if args[2] == address:
                results.append((args[1], args[3]))
        return results

    def test_torque_disabled_before_config_writes(self, mock_sdk: MagicMock) -> None:
        """Torque is disabled before any configuration registers are written."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        calls_1byte, _ = self._get_write_calls(mock_sdk)

        # First 6 write1ByteTxRx calls should be torque disable (addr=40, value=0)
        first_six = [(c.args[2], c.args[3]) for c in calls_1byte[:6]]
        assert all(addr == 40 and val == 0 for addr, val in first_six)

    def test_operating_mode_set_to_position(self, mock_sdk: MagicMock) -> None:
        """Operating mode is set to position control (0) for all servos."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        calls_1byte, _ = self._get_write_calls(mock_sdk)
        mode_writes = self._extract_register_writes(calls_1byte, 33)

        assert len(mode_writes) == 6
        assert all(val == 0 for _, val in mode_writes)

    def test_gripper_protection_written_only_for_gripper(self, mock_sdk: MagicMock) -> None:
        """Gripper protection registers are written only to the gripper servo (ID 6)."""
        robot = _create_robot(mock_sdk, role="follower")
        robot.connect()

        calls_1byte, calls_2byte = self._get_write_calls(mock_sdk)

        # 2-byte writes: MAX_TORQUE_LIMIT (addr 16) and PROTECTION_CURRENT (addr 28)
        torque_limit_writes = self._extract_register_writes(calls_2byte, 16)
        current_writes = self._extract_register_writes(calls_2byte, 28)
        # 1-byte write: OVERLOAD_TORQUE (addr 36)
        overload_writes = self._extract_register_writes(calls_1byte, 36)

        assert len(torque_limit_writes) == 1
        assert torque_limit_writes[0] == (6, 500)
        assert len(current_writes) == 1
        assert current_writes[0] == (6, 250)
        assert len(overload_writes) == 1
        assert overload_writes[0] == (6, 25)


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
        """Calibrated observation returns normalized values by default."""
        robot = _create_robot(mock_sdk, calibration=calibration_obj)
        robot.connect()
        obs = robot.get_observation()

        # All servos return 2048 ticks.
        state = obs.joint_positions
        assert state.shape == (6,)
        assert state.dtype == np.float32
        # shoulder_pan and shoulder_lift are mapped by calibration range.
        assert state[0] == pytest.approx(-1.83, abs=0.1)
        assert state[1] == pytest.approx(5.15, abs=0.1)


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
        # Sending a large normalized value should be clamped internally.
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

        from physicalai.robot.so101.calibration import SO101Calibration

        with pytest.raises(ValueError, match="missing joints"):
            SO101Calibration.from_path(path)

    def test_from_path_not_a_dict(self, tmp_path: Path) -> None:
        """Calibration file that is a JSON array (not a dict) raises TypeError."""
        path = tmp_path / "bad_format.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        from physicalai.robot.so101.calibration import SO101Calibration

        with pytest.raises(TypeError, match="JSON object"):
            SO101Calibration.from_path(path)

    def test_from_path_bad_drive_mode(self, tmp_path: Path) -> None:
        """Calibration with drive_mode not in {0, 1} raises ValueError."""
        bad_cal = {
            name: {**v, "drive_mode": 2} if name == "shoulder_pan" else v for name, v in SAMPLE_CALIBRATION.items()
        }
        path = tmp_path / "bad_drive_mode.json"
        path.write_text(json.dumps(bad_cal), encoding="utf-8")

        from physicalai.robot.so101.calibration import SO101Calibration

        with pytest.raises(ValueError, match="drive_mode must be 0 or 1"):
            SO101Calibration.from_path(path)

    def test_tick_normalized_roundtrip(self, mock_sdk: MagicMock, calibration_obj: Any) -> None:
        """Converting ticks → normalized → ticks should roundtrip."""
        from physicalai.robot.so101 import SO101

        robot = SO101(port="/dev/ttyUSB0", role="follower", calibration=calibration_obj, unit="normalized")
        original_ticks = np.array([2500, 1400, 2100, 1900, 2600, 2300], dtype=np.int32)
        normalized = robot._ticks_to_unit(original_ticks)  # noqa: SLF001
        recovered_ticks = robot._unit_to_ticks(normalized)  # noqa: SLF001
        np.testing.assert_array_almost_equal(original_ticks, recovered_ticks, decimal=0)
