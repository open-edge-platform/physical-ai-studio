# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SO-101 visual identification routine.

``identify_so101_robot_visually`` drives the physicalai ``SO101`` driver in
raw-ticks (uncalibrated) mode and wiggles the gripper a small amount around its
current position. The critical regression this guards is driving a joint to a
physical stop, which can stall an STS3215 servo into overload protection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import robots.catalog.so101 as so101_module
from exceptions import RobotIdentifyError
from robots.catalog.so101 import SO101Probe, SO101RobotPayload
from schemas import SerialPortInfo

if TYPE_CHECKING:
    from collections.abc import Generator

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PORT = "/dev/ttyACM0"
SERIAL_NUMBER = "SO101-2026-0001"


class _FakeDriver:
    """Stands in for physicalai's SO101 driver."""

    JOINT_ORDER = list(JOINT_NAMES)
    instances: list[_FakeDriver] = []
    default_start: np.ndarray = np.zeros(6, dtype=np.float32)
    default_observation_failure: Exception | None = None

    def __init__(
        self,
        *,
        port: str | None = None,
        role: str | None = None,
        unit: str | None = None,
    ) -> None:
        self.port = port
        self.role = role
        self.unit = unit
        self.current_position = _FakeDriver.default_start.copy()
        self._observation_failure = _FakeDriver.default_observation_failure
        self.actions: list[np.ndarray] = []
        self._torque_on_disconnect = True
        self.torque_commands: list[bool] = []
        self.connected = False
        self.disconnected = False
        _FakeDriver.instances.append(self)

    @classmethod
    def uncalibrated(
        cls,
        port: str,
        baudrate: int = 1_000_000,
        role: str = "follower",
        unit: str = "ticks",
    ) -> _FakeDriver:
        return cls(port=port, role=role, unit=unit)

    @property
    def torque_on_disconnect(self) -> bool:
        return self._torque_on_disconnect

    @torque_on_disconnect.setter
    def torque_on_disconnect(self, value: bool) -> None:
        self._torque_on_disconnect = value

    def connect(self) -> None:
        self.connected = True

    def get_observation(self) -> SimpleNamespace:
        if self._observation_failure is not None:
            raise self._observation_failure
        return SimpleNamespace(joint_positions=self.current_position.copy())

    def send_action(self, action: np.ndarray) -> None:
        self.actions.append(action.copy())
        self.current_position = action.copy()

    def set_torque(self, *, enabled: bool) -> None:
        self.torque_commands.append(enabled)

    def disconnect(self) -> None:
        self.disconnected = True


class _FakePortScanner:
    def __init__(self, robots: list[SerialPortInfo] | None = None) -> None:
        self.robots = (
            [SerialPortInfo(connection_string=PORT, serial_number=SERIAL_NUMBER)] if robots is None else robots
        )

    async def find_robots(self) -> list[SerialPortInfo]:
        return self.robots


def _payload() -> SO101RobotPayload:
    return SO101RobotPayload(connection_string="", serial_number=SERIAL_NUMBER)


@pytest.fixture(autouse=True)
def _clear_fake_drivers() -> Generator[None, None, None]:
    _FakeDriver.instances.clear()
    _FakeDriver.default_start = np.zeros(6, dtype=np.float32)
    _FakeDriver.default_observation_failure = None
    yield
    _FakeDriver.instances.clear()
    _FakeDriver.default_start = np.zeros(6, dtype=np.float32)
    _FakeDriver.default_observation_failure = None


def _last_driver() -> _FakeDriver:
    return _FakeDriver.instances[-1]


@pytest.mark.asyncio
async def test_identify_uses_uncalibrated_driver_with_safe_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(so101_module, "SO101", _FakeDriver)

    start = np.zeros(6, dtype=np.float32)
    start[5] = 2048.0
    _FakeDriver.default_start = start

    await SO101Probe().identify(_payload(), _FakePortScanner())

    driver = _last_driver()
    assert driver.unit == "ticks"
    assert driver.role == "follower"
    assert driver.connected
    assert driver.disconnected
    assert driver.torque_on_disconnect is False
    assert driver.torque_commands == [False]
    assert len(driver.actions) == 3

    gripper_goals = [float(action[5]) for action in driver.actions]
    assert all(0.0 <= goal <= 4095.0 for goal in gripper_goals)
    assert gripper_goals[-1] == 2048.0


@pytest.mark.asyncio
async def test_identify_lets_connect_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingDriver(_FakeDriver):
        def connect(self) -> None:
            raise ConnectionError(f"Failed to open serial port {PORT}")

    monkeypatch.setattr(so101_module, "SO101", _FailingDriver)

    # Port open / permission failures keep the standard serial error mapping
    # instead of the identify (power-cycle) error.
    with pytest.raises(ConnectionError, match="Failed to open serial port"):
        await SO101Probe().identify(_payload(), _FakePortScanner())

    # physicalai cleans up the port itself when connect fails; torque is never
    # touched because the motion never started.
    assert not _last_driver().disconnected
    assert _last_driver().torque_commands == []


@pytest.mark.asyncio
async def test_identify_wraps_overload_as_robot_identify_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(so101_module, "SO101", _FakeDriver)
    _FakeDriver.default_start = np.array([0, 0, 0, 0, 0, 2048.0], dtype=np.float32)
    _FakeDriver.default_observation_failure = ConnectionError("Servo 'gripper' (ID 6) data not available in sync read")

    with pytest.raises(RobotIdentifyError, match="overload"):
        await SO101Probe().identify(_payload(), _FakePortScanner())

    driver = _last_driver()
    assert driver.disconnected
    assert driver.torque_commands == [False]


@pytest.mark.asyncio
async def test_identify_raises_when_port_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(so101_module, "SO101", _FakeDriver)

    with pytest.raises(ValueError, match="serial port"):
        await SO101Probe().identify(_payload(), _FakePortScanner(robots=[]))
