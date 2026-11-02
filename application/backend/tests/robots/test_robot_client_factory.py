# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for RobotClientFactory, which owns SharedRobot creation.

Catalog builders return a plain physicalai driver; the factory is the single
place that wraps it in a ``SharedRobot`` and names the owner. These tests cover
that seam.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from physicalai.robot import SharedRobot

from robots.robot_client_factory import RobotClientFactory
from schemas import SerialPortInfo
from schemas.robot import RobotAdapter

# Free-form user text. SharedRobot names key Zenoh topics and only accept
# letters, digits, '_' and '-', so this must never reach the transport.
DISPLAY_NAME = "My SO101 Arm #1"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class _FakeConnectionManager:
    """Stands in for RobotConnectionManager with one discovered SO101 port."""

    def __init__(self, *, serial_number: str | None = "ABC123") -> None:
        self.robots = [SerialPortInfo(connection_string="/dev/ttyACM0", serial_number=serial_number)]

    async def find_robots(self) -> list[SerialPortInfo]:
        return self.robots

    async def find_port(self, port_info: SerialPortInfo) -> str | None:
        for port in self.robots:
            if port_info.serial_number and port.serial_number == port_info.serial_number:
                return port.connection_string
            if not port_info.serial_number and port.connection_string == port_info.connection_string:
                return port.connection_string
        return None


def _calibration() -> dict[str, dict[str, int]]:
    return {
        name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(JOINT_NAMES)
    }


def _robot(robot_type: str, payload: dict[str, Any], *, robot_id: UUID | None = None) -> Any:
    return RobotAdapter.validate_python(
        {
            "id": str(robot_id or uuid4()),
            "name": DISPLAY_NAME,
            "type": robot_type,
            "payload": payload,
        }
    )


def _so101_robot(robot_type: str = "SO101_Follower", *, robot_id: UUID | None = None) -> Any:
    return _robot(
        robot_type,
        {"connection_string": "/dev/ttyACM0", "serial_number": "ABC123", "calibration": _calibration()},
        robot_id=robot_id,
    )


def _factory() -> RobotClientFactory:
    return RobotClientFactory(robot_manager=_FakeConnectionManager())


class TestBuild:
    async def test_build_shared_robot_returns_transport_and_definition(self) -> None:
        robot = _so101_robot()

        shared, definition = await _factory().build_shared_robot(robot)

        assert isinstance(shared, SharedRobot)
        assert shared.name == str(robot.id)
        assert definition.type == "SO101_Follower"


class TestFindPort:
    """Port resolution is shared by runtime config export and calibration."""

    async def test_a_discovered_port_is_reported_as_its_stable_alias(self, mocker: Any) -> None:
        mocker.patch(
            "robots.robot_client_factory.resolve_serial_device",
            return_value="/dev/serial/by-id/usb-test",
        )

        port = await _factory().find_port(SerialPortInfo(connection_string="/dev/ttyACM0", serial_number="ABC123"))

        assert port == "/dev/serial/by-id/usb-test"

    async def test_a_robot_without_a_serial_number_keeps_the_raw_path(self, mocker: Any) -> None:
        """Matching by path locates a device without identifying it.

        A by-id alias would make that guess look verified, and in an exported
        config it would hide the path behind a name that no longer needs a
        CHANGE_ME marker.
        """
        resolve = mocker.patch("robots.robot_client_factory.resolve_serial_device")
        factory = RobotClientFactory(robot_manager=_FakeConnectionManager(serial_number=None))  # type: ignore[arg-type]

        port = await factory.find_port(SerialPortInfo(connection_string="/dev/ttyACM0", serial_number=None))

        assert port == "/dev/ttyACM0"
        resolve.assert_not_called()

    async def test_an_absent_robot_resolves_to_nothing(self) -> None:
        """Not falling back to the stored port is deliberate: it may be another robot by now."""
        port = await _factory().find_port(SerialPortInfo(connection_string="/dev/ttyACM0", serial_number="GONE"))

        assert port is None
