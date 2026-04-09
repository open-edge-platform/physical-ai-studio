# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO-101 robot arm driver.

Concrete implementation of the :class:`~physicalai.robot.protocol.Robot` protocol
for the SO-101 robot arm (6-DOF, Feetech STS3215 servos).

Requires the ``feetech-servo-sdk`` package::

    pip install physicalai[so101]

The driver supports two roles:

* **follower** (default) — torque enabled, used for inference / deployment.
* **leader** — torque disabled, used for teleoperation (read-only).

Calibration data can be loaded from a JSON file so that joint positions are
reported in radians rather than raw servo ticks.

By default, this driver requires calibration and uses radians for both state
and action. A dedicated :meth:`SO101.uncalibrated` factory exists for explicit
raw-ticks bringup/debug mode.
"""

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from loguru import logger

from physicalai.robot import Robot
from physicalai.robot.so101.calibration import SO101Calibration
from physicalai.robot.so101.constants import (
    PROTOCOL_VERSION,
    RADIANS_PER_TICK,
    SO101_JOINT_ORDER,
    TICKS_PER_REVOLUTION,
    VALID_ROLES,
    STS3215Addr,
    STS3215Len,
)


@dataclass(frozen=True)
class _SO101Connection:
    """Active connection state for the SO-101 serial bus."""

    port_handler: Any
    packet_handler: Any
    group_sync_read: Any
    group_sync_write: Any


class SO101(Robot):
    """Driver for the SO-101 robot arm (6-DOF, Feetech STS3215 servos).

    Args:
        port: Serial port path, e.g. ``"/dev/ttyUSB0"`` or ``"/dev/ttyACM0"``.
        baudrate: Serial baudrate. Defaults to 1 000 000 (STS3215 factory default).
        role: ``"follower"`` (torque enabled, full control) or ``"leader"``
            (torque disabled, read-only for teleoperation).
        calibration: SO-101 calibration object or calibration JSON path.
            This is required for normal operation and defines the robot
            coordinate frame (radians).
    """

    JOINT_ORDER: ClassVar[list[str]] = list(SO101_JOINT_ORDER)
    """Canonical joint ordering (index 0 → first element of state vector)."""

    NUM_JOINTS: ClassVar[int] = 6
    """Number of joints / servos on the SO-101."""

    def __init__(
        self,
        port: str,
        calibration: SO101Calibration | str | Path | None,
        baudrate: int = 1_000_000,
        role: Literal["leader", "follower"] = "follower",
        *,
        _allow_uncalibrated: bool = False,  # must be passed by keyword
    ) -> None:
        """Initialize the SO-101 driver (does not open the connection).

        ``calibration`` may be:

        * ``SO101Calibration`` — use an already loaded calibration object.
        * ``str | Path`` — load LeRobot calibration JSON from disk.
        * ``None`` — only allowed via :meth:`SO101.uncalibrated` for raw ticks.

        Raises:
            ValueError: If ``role`` is not ``"leader"`` or ``"follower"``.
            ValueError: If resolved servo IDs are invalid.
        """
        if role not in VALID_ROLES:
            msg = f"Invalid role {role!r}. Must be one of {sorted(VALID_ROLES)}."
            raise ValueError(msg)

        self._port = port
        self._baudrate = baudrate
        self._role = role

        # Calibration -------------------------------------------------------
        if calibration is None and not _allow_uncalibrated:
            msg = (
                "calibration is required for SO101. "
                "Pass a calibration object/path, or use SO101.uncalibrated(...) "
                "for explicit raw-ticks bringup mode."
            )
            raise ValueError(msg)

        if isinstance(calibration, (str, Path)):
            calibration = SO101Calibration.from_path(calibration)

        self._calibration: SO101Calibration | None = calibration
        self._uncalibrated_mode = self._calibration is None
        self._warned_uncalibrated = False
        if self._calibration is not None:
            self.servo_ids = {name: self._calibration.joints[name].id for name in self.JOINT_ORDER}
        else:
            # Explicit uncalibrated mode fallback — assumes canonical 1..6 mapping.
            self.servo_ids = {name: idx for idx, name in enumerate(self.JOINT_ORDER, 1)}

        # Connection state (set during connect()) --------------------------
        self._connection: _SO101Connection | None = None

        # Torque ON/OFF behavior on disconnect (default: True for follower, False for leader)
        self._torque_on_disconnect: bool = role == "follower"

    @classmethod
    def uncalibrated(
        cls,
        port: str,
        baudrate: int = 1_000_000,
        role: Literal["leader", "follower"] = "follower",
    ) -> "SO101":
        """Create an SO-101 instance in explicit raw-ticks mode.

        This mode is intended for bringup/debug only. Observations and actions
        use raw servo ticks (0-4095), not radians.

        Warning:
            Uncalibrated mode is not suitable for policy inference/deployment.

        Returns:
            Uncalibrated SO101 instance.
        """
        return cls(
            port=port,
            calibration=None,
            baudrate=baudrate,
            role=role,
            _allow_uncalibrated=True,
        )

    @property
    def port(self) -> str:
        """Serial port path."""
        return self._port

    @port.setter
    def port(self, value: str) -> None:
        self._port = value

    @property
    def baudrate(self) -> int:
        """Serial baudrate."""
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value: int) -> None:
        if value <= 0:
            msg = f"baudrate must be a positive integer, got {value!r}"
            raise ValueError(msg)
        self._baudrate = value

    @property
    def role(self) -> Literal["leader", "follower"]:
        """Robot role: ``"follower"`` or ``"leader"``."""
        return self._role

    @role.setter
    def role(self, value: Literal["leader", "follower"]) -> None:
        if value not in VALID_ROLES:
            msg = f"Invalid role {value!r}. Must be one of {sorted(VALID_ROLES)}."
            raise ValueError(msg)
        self._role = value

    @property
    def calibrated(self) -> bool:
        """Whether this driver is running with calibration (radian mode)."""
        return self._calibration is not None

    @property
    def unit(self) -> str:
        """Current state/action unit: ``"radians"`` or ``"ticks"``."""
        return "radians" if self.calibrated else "ticks"

    def _require_connection(self) -> _SO101Connection:
        """Return the active connection, or raise if disconnected.

        Returns:
            The active connection state.

        Raises:
            ConnectionError: If :meth:`connect` has not been called.
        """
        conn = self._connection
        if conn is None:
            msg = "Robot is not connected. Call connect() first."
            raise ConnectionError(msg)
        return conn

    def _require_calibration(self) -> SO101Calibration:
        """Return calibration data, or raise if running in uncalibrated mode.

        Returns:
            The loaded SO-101 calibration data.

        Raises:
            RuntimeError: If calibration data is unavailable.
        """
        if self._calibration is None:
            msg = (
                "Calibration is required for tick/radian conversion. "
                "Provide calibration or avoid conversion methods in uncalibrated mode."
            )
            raise RuntimeError(msg)
        return self._calibration

    def connect(self) -> None:
        """Open the serial port, ping all servos, and configure torque.

        Idempotent: calling ``connect()`` on an already-connected robot is a
        no-op.

        Raises:
            ImportError: If ``feetech-servo-sdk`` is not installed.
            ConnectionError: If the serial port cannot be opened or a servo
                does not respond to ping.
        """
        if self._connection is not None:
            return  # already connected

        # Lazy import — only pull in the SDK when actually connecting.
        try:
            from scservo_sdk import (  # type: ignore[import-untyped]  # noqa: PLC0415
                GroupSyncRead,
                GroupSyncWrite,
                PacketHandler,
                PortHandler,
            )
        except ImportError:
            msg = "feetech-servo-sdk is required for SO-101 support. Install it with:  pip install physicalai[so101]"
            raise ImportError(msg) from None

        # Open port ---------------------------------------------------------
        port_handler = PortHandler(self.port)
        if not port_handler.openPort():
            msg = f"Failed to open serial port {self.port}"
            raise ConnectionError(msg)

        # Set a packet timeout so pings/reads don't block forever.
        port_handler.setPacketTimeoutMillis(50.0)

        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            msg = f"Failed to set baudrate {self.baudrate} on {self.port}"
            raise ConnectionError(msg)

        packet_handler = PacketHandler(PROTOCOL_VERSION)

        try:
            # Sync read / write groups -----------------------------------------
            group_sync_read = GroupSyncRead(
                port_handler,
                packet_handler,
                STS3215Addr.PRESENT_POSITION,
                STS3215Len.PRESENT_POSITION,
            )
            for servo_id in self.servo_ids.values():
                if not group_sync_read.addParam(servo_id):
                    msg = f"Failed to add servo {servo_id} to sync read group"
                    raise ConnectionError(msg)  # noqa: TRY301

            group_sync_write = GroupSyncWrite(
                port_handler,
                packet_handler,
                STS3215Addr.GOAL_POSITION,
                STS3215Len.GOAL_POSITION,
            )

            self._connection = _SO101Connection(
                port_handler=port_handler,
                packet_handler=packet_handler,
                group_sync_read=group_sync_read,
                group_sync_write=group_sync_write,
            )

            # Ping all servos ---------------------------------------------------
            self._ping_servos()

            # Configure torque based on role ------------------------------------
            self._set_torque(enabled=self.role == "follower")
        except Exception:
            with contextlib.suppress(Exception):
                port_handler.closePort()
            self._connection = None
            raise

        logger.info(f"SO-101 connected on {self.port} (role={self.role}, servos={self.servo_ids})")

    @property
    def torque_on_disconnect(self) -> bool:
        """Whether torque remains enabled after disconnect."""
        return self._torque_on_disconnect

    @torque_on_disconnect.setter
    def torque_on_disconnect(self, value: bool) -> None:
        """Set whether torque should be enabled on disconnect.

        Skips the hold-position safety behavior if torque is disabled.

        Warning:
            The arm will drop under gravity. Only use when the arm is
            in a safe position or manually supported.

        Raises:
            ValueError: if setting torque on for non-follower robot.
        """
        if self.role != "follower" and value:
            msg = "Torque on disconnect can only be enabled for follower arms."
            raise ValueError(msg)
        if not value and self._torque_on_disconnect:
            logger.warning(
                "Disabling torque on disconnect will cause the arm to drop under gravity. Ensure this is intentional.",
            )
        self._torque_on_disconnect = value

    def disconnect(self) -> None:
        """Disconnect from the robot, leaving it in a safe state.

        * **Follower**: torque remains enabled (arm holds position).
        * **Leader**: torque stays disabled.

        Idempotent: calling ``disconnect()`` when not connected is a no-op.
        """
        conn = self._connection
        if conn is None:
            return  # not connected

        try:
            if self._torque_on_disconnect:
                self._hold_position()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to hold position while disconnecting SO-101; proceeding to close port.",
            )
        finally:
            self._connection = None
            try:
                conn.port_handler.closePort()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Error while closing SO-101 serial port; continuing cleanup.",
                )

        logger.info(f"SO-101 disconnected from {self.port}")

    def get_observation(self) -> dict[str, Any]:
        """Read current joint positions from all servos.

        Returns:
            A dict with:

            * ``"state"``: ``np.ndarray`` of shape ``(6,)`` — joint positions
                            in radians by default, or raw ticks in explicit uncalibrated mode.
            * ``"timestamp"``: ``float`` from ``time.monotonic()``.
        """
        raw_positions = self._read_joint_positions()

        if self._calibration is not None:
            state = self._ticks_to_radians(raw_positions)
        else:
            if not self._warned_uncalibrated:
                logger.warning(
                    "SO101 running in explicit uncalibrated mode. Joint "
                    "positions/actions are raw servo ticks (0-4095), not radians. "
                    "Do not use uncalibrated mode for policy inference/deployment.",
                )
                self._warned_uncalibrated = True
            state = raw_positions.astype(np.float32)

        return {
            "state": state,
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        """Send joint position commands to all servos.

        Args:
            action: Array of shape ``(6,)`` with target joint positions in
                radians by default, or raw ticks in explicit uncalibrated mode.

        Raises:
            RuntimeError: If the robot is in ``"leader"`` role.
            ValueError: If the action shape does not match ``(6,)``.
        """
        if self.role == "leader":
            msg = "Cannot send actions to a leader arm. Leader arms are read-only for teleoperation."
            raise RuntimeError(msg)

        expected_shape = (self.NUM_JOINTS,)
        if action.shape != expected_shape:
            msg = f"Expected action shape {expected_shape}, got {action.shape}"
            raise ValueError(msg)

        ticks = self._radians_to_ticks(action) if self._calibration is not None else np.round(action).astype(np.int32)
        self._write_joint_positions(ticks)

    def _ticks_to_radians(self, ticks: np.ndarray) -> np.ndarray:
        """Convert raw servo ticks to radians using calibration data.

        Args:
            ticks: Integer tick values, shape ``(6,)``.

        Returns:
            Float32 array of joint positions in radians, shape ``(6,)``.
        """
        calibration = self._require_calibration()
        result = np.empty(self.NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(self.JOINT_ORDER):
            cal = calibration.joints[name]
            result[i] = (ticks[i] - cal.homing_offset) * cal.direction * RADIANS_PER_TICK
        return result

    def _radians_to_ticks(self, radians: np.ndarray) -> np.ndarray:
        """Convert radians to raw servo ticks, clamping to calibration range.

        Args:
            radians: Float joint positions in radians, shape ``(6,)``.

        Returns:
            Int32 array of tick values, shape ``(6,)``.
        """
        calibration = self._require_calibration()
        result = np.empty(self.NUM_JOINTS, dtype=np.int32)
        for i, name in enumerate(self.JOINT_ORDER):
            cal = calibration.joints[name]
            ticks_val = round(radians[i] / (cal.direction * RADIANS_PER_TICK) + cal.homing_offset)
            result[i] = int(np.clip(ticks_val, cal.range_min, cal.range_max))
        return result

    def _ping_servos(self) -> None:
        """Ping every servo and raise on failure.

        Raises:
            ConnectionError: If a servo does not respond.
        """
        conn = self._require_connection()

        for name, servo_id in self.servo_ids.items():
            _, comm_result, error = conn.packet_handler.ping(conn.port_handler, servo_id)
            if comm_result != 0:
                msg = f"Servo '{name}' (ID {servo_id}) did not respond on {self.port}. Comm result: {comm_result}"
                raise ConnectionError(msg)
            if error != 0:
                logger.warning(f"Servo '{name}' (ID {servo_id}) returned error: {error}")

    def _set_torque(self, *, enabled: bool) -> None:
        """Enable or disable torque on all servos."""
        conn = self._require_connection()

        value = 1 if enabled else 0
        for name, servo_id in self.servo_ids.items():
            comm_result, error = conn.packet_handler.write1ByteTxRx(
                conn.port_handler,
                servo_id,
                STS3215Addr.TORQUE_ENABLE,
                value,
            )
            if comm_result != 0:
                logger.warning(f"Failed to set torque on servo '{name}' (ID {servo_id}): comm={comm_result}")
            if error != 0:
                logger.warning(f"Torque write error on servo '{name}' (ID {servo_id}): err={error}")

    def _hold_position(self) -> None:
        """Command all servos to hold their current position.

        Reads the current positions and writes them back as goal positions,
        then ensures torque is enabled.  This prevents the arm from dropping
        under gravity when the connection is closed.
        """
        raw = self._read_joint_positions()
        self._write_joint_positions(raw.astype(np.int32))
        self._set_torque(enabled=True)

    def _read_joint_positions(self) -> np.ndarray:
        """Bulk-read present positions from all servos via sync read.

        Returns:
            Int32 array of raw tick positions, shape ``(6,)``.

        Raises:
            ConnectionError: If sync read fails.
        """
        conn = self._require_connection()

        comm_result = conn.group_sync_read.txRxPacket()
        if comm_result != 0:
            msg = f"Sync read failed with comm result {comm_result}"
            raise ConnectionError(msg)

        positions = np.empty(self.NUM_JOINTS, dtype=np.int32)
        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = self.servo_ids[name]
            if not conn.group_sync_read.isAvailable(
                servo_id,
                STS3215Addr.PRESENT_POSITION,
                STS3215Len.PRESENT_POSITION,
            ):
                msg = f"Servo '{name}' (ID {servo_id}) data not available in sync read"
                raise ConnectionError(msg)
            positions[i] = conn.group_sync_read.getData(
                servo_id,
                STS3215Addr.PRESENT_POSITION,
                STS3215Len.PRESENT_POSITION,
            )
        return positions

    def _write_joint_positions(self, ticks: np.ndarray) -> None:
        """Bulk-write goal positions to all servos via sync write.

        Args:
            ticks: Int32 array of goal tick positions, shape ``(6,)``.

        Raises:
            ConnectionError: If sync write fails.
        """
        conn = self._require_connection()

        # Hardware safety clamp — STS3215 valid range is 0..4095
        ticks = np.clip(ticks, 0, TICKS_PER_REVOLUTION - 1)

        conn.group_sync_write.clearParam()

        for i, name in enumerate(self.JOINT_ORDER):
            servo_id = self.servo_ids[name]
            position = int(ticks[i])
            # STS3215 goal position is 2 bytes, little-endian
            param = [position & 0xFF, (position >> 8) & 0xFF]
            if not conn.group_sync_write.addParam(servo_id, param):
                msg = f"Failed to add servo '{name}' (ID {servo_id}) to sync write"
                raise ConnectionError(msg)

        comm_result = conn.group_sync_write.txPacket()
        if comm_result != 0:
            msg = f"Sync write failed with comm result {comm_result}"
            raise ConnectionError(msg)
