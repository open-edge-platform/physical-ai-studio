# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catalog robot builders that produce SharedRobot drivers.

Robots are built through ``RobotAdapter``, the same discriminated-union adapter
the API deserializes with, so these tests exercise the dynamically generated
per-type models (``SO101_FollowerRobot``) that builders actually receive rather
than the hand-written convenience models.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from physicalai.config import instantiate, to_config

from robots.catalog.so101 import _build_so101_driver
from robots.catalog.widowxai import _build_trossen_bimanual_driver, _build_trossen_single_arm_driver
from schemas.robot import RobotAdapter

# Free-form user text: what a real operator types, and what the Zenoh
# transport rejects as a topic key.
DISPLAY_NAME = "My SO101 Arm #1"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _calibration() -> dict[str, dict[str, int]]:
    return {
        name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(JOINT_NAMES)
    }


def _robot(robot_type: str, payload: dict[str, Any], *, robot_id: UUID | None = None) -> Any:
    """Build a robot exactly as the API does, via the registry-backed adapter."""
    return RobotAdapter.validate_python(
        {
            "id": str(robot_id or uuid4()),
            "name": DISPLAY_NAME,
            "type": robot_type,
            "payload": payload,
        }
    )


class _StubFactory:
    """Minimal CatalogRobotFactory: resolves any request to a fixed port."""

    def __init__(self, port: str | None = "/dev/ttyACM0") -> None:
        self._port = port

    async def find_port(self, _port_info) -> str | None:
        return self._port


def _so101_robot(robot_type: str = "SO101_Follower", *, robot_id: UUID | None = None, calibrated: bool = True) -> Any:
    return _robot(
        robot_type,
        {
            "connection_string": "/dev/ttyACM0",
            "serial_number": "",
            "calibration": _calibration() if calibrated else None,
        },
        robot_id=robot_id,
    )


def _driver_recipe(shared) -> dict[str, Any]:
    """The nested driver ComponentConfig the owner process rebuilds."""
    return to_config(shared)["init_args"]["robot"]


class TestRegistryModelShape:
    """Guards the assumption these tests and the builders depend on."""

    def test_builder_receives_a_generated_model_not_the_local_one(self) -> None:
        from robots.catalog.so101 import SO101Robot

        robot = _so101_robot("SO101_Leader")

        assert type(robot).__name__ == "SO101_LeaderRobot"
        # The historical bug: builders narrowed on SO101Robot, which never matches.
        assert not isinstance(robot, SO101Robot)


class TestSO101Builder:
    async def test_builds_from_the_generated_model(self) -> None:
        shared = await _build_so101_driver(_so101_robot(), _StubFactory())

        assert _driver_recipe(shared)["class_path"] == "physicalai.robot.SO101"

    async def test_uses_transport_safe_name_not_display_name(self) -> None:
        robot_id = uuid4()
        shared = await _build_so101_driver(_so101_robot(robot_id=robot_id), _StubFactory())

        # The display name would be rejected by the transport, so the id is used.
        assert shared.name == str(robot_id)

    async def test_exports_driver_recipe_for_the_owner(self) -> None:
        shared = await _build_so101_driver(_so101_robot(), _StubFactory())

        init_args = _driver_recipe(shared)["init_args"]
        assert init_args["port"] == "/dev/ttyACM0"
        assert init_args["role"] == "follower"
        assert init_args["unit"] == "normalized"
        assert set(init_args["calibration"]) == set(JOINT_NAMES)

    async def test_recipe_rebuilds_a_calibrated_driver(self) -> None:
        """The owner process instantiates the recipe; it must yield a usable driver."""
        shared = await _build_so101_driver(_so101_robot(), _StubFactory())

        driver = instantiate(_driver_recipe(shared))
        assert driver.joint_names == list(JOINT_NAMES)

    async def test_leader_role_is_exported(self) -> None:
        shared = await _build_so101_driver(_so101_robot("SO101_Leader"), _StubFactory())

        assert _driver_recipe(shared)["init_args"]["role"] == "leader"

    async def test_missing_calibration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="calibration is required"):
            await _build_so101_driver(_so101_robot(calibrated=False), _StubFactory())

    async def test_unresolvable_port_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Could not resolve a serial port"):
            await _build_so101_driver(_so101_robot(), _StubFactory(port=None))

    async def test_wrong_payload_type_is_rejected(self) -> None:
        wrong = _robot("Trossen_WidowXAI_Follower", {"connection_string": "192.168.1.2"})

        with pytest.raises(TypeError, match="Expected SO101RobotPayload"):
            await _build_so101_driver(wrong, _StubFactory())


class TestTrossenBuilders:
    async def test_single_arm_builds_from_the_generated_model(self) -> None:
        robot_id = uuid4()
        robot = _robot("Trossen_WidowXAI_Follower", {"connection_string": "192.168.1.2"}, robot_id=robot_id)

        shared = await _build_trossen_single_arm_driver(robot, _StubFactory())

        assert shared.name == str(robot_id)
        assert _driver_recipe(shared) == {
            "class_path": "physicalai.robot.WidowXAI",
            "init_args": {"ip": "192.168.1.2", "role": "follower"},
        }

    async def test_single_arm_leader_role_is_exported(self) -> None:
        robot = _robot("Trossen_WidowXAI_Leader", {"connection_string": "192.168.1.2"})

        shared = await _build_trossen_single_arm_driver(robot, _StubFactory())

        assert _driver_recipe(shared)["init_args"]["role"] == "leader"

    async def test_bimanual_is_owned_as_one_robot(self) -> None:
        robot_id = uuid4()
        robot = _robot(
            "Trossen_Bimanual_WidowXAI_Follower",
            {"connection_string_left": "192.168.1.2", "connection_string_right": "192.168.1.3"},
            robot_id=robot_id,
        )

        shared = await _build_trossen_bimanual_driver(robot, _StubFactory())

        assert shared.name == str(robot_id)
        # A single owner holds both arms, rather than one SharedRobot per arm.
        recipe = _driver_recipe(shared)
        assert recipe["class_path"] == "physicalai.robot.BimanualWidowXAI"
        assert recipe["init_args"]["left"]["init_args"]["ip"] == "192.168.1.2"
        assert recipe["init_args"]["right"]["init_args"]["ip"] == "192.168.1.3"

    async def test_bimanual_recipe_rebuilds_both_arms(self) -> None:
        robot = _robot(
            "Trossen_Bimanual_WidowXAI_Leader",
            {"connection_string_left": "192.168.1.2", "connection_string_right": "192.168.1.3"},
        )

        shared = await _build_trossen_bimanual_driver(robot, _StubFactory())

        joint_names = instantiate(_driver_recipe(shared)).joint_names
        assert any(name.startswith("left_") for name in joint_names)
        assert any(name.startswith("right_") for name in joint_names)

    async def test_wrong_payload_type_is_rejected(self) -> None:
        wrong = _so101_robot()

        with pytest.raises(TypeError, match="Expected TrossenSingleArmPayload"):
            await _build_trossen_single_arm_driver(wrong, _StubFactory())
