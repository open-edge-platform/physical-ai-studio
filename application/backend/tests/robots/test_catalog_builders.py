# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the catalog robot builders that produce SharedRobot drivers."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from physicalai.config import instantiate, to_config
from physicalai.robot.so101 import SO101JointCalibration

from robots.catalog.so101 import SO101Robot, SO101RobotPayload, _build_so101_driver
from robots.catalog.widowxai import (
    TrossenBimanualPayload,
    TrossenBimanualRobot,
    TrossenSingleArmPayload,
    TrossenSingleArmRobot,
    _build_trossen_bimanual_driver,
    _build_trossen_single_arm_driver,
)

# Free-form user text: what a real operator types, and what the Zenoh
# transport rejects as a topic key.
DISPLAY_NAME = "My SO101 Arm #1"

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _calibration() -> dict[str, SO101JointCalibration]:
    return {
        name: SO101JointCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
        for i, name in enumerate(JOINT_NAMES)
    }


class _StubFactory:
    """Minimal CatalogRobotFactory: resolves any request to a fixed port."""

    def __init__(self, port: str | None = "/dev/ttyACM0") -> None:
        self._port = port

    async def find_port(self, _port_info) -> str | None:
        return self._port


def _so101_robot(robot_id: UUID, *, calibration: dict[str, SO101JointCalibration] | None = None) -> SO101Robot:
    return SO101Robot(
        id=robot_id,
        name=DISPLAY_NAME,
        type="SO101_Follower",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        payload=SO101RobotPayload(
            connection_string="/dev/ttyACM0",
            serial_number="",
            calibration=_calibration() if calibration is None else calibration,
        ),
    )


class TestSO101Builder:
    async def test_uses_transport_safe_name_not_display_name(self) -> None:
        robot_id = uuid4()
        shared = await _build_so101_driver(_so101_robot(robot_id), _StubFactory())

        # The display name would be rejected by the transport, so the id is used.
        assert shared.name == str(robot_id)

    async def test_exports_driver_recipe_for_the_owner(self) -> None:
        shared = await _build_so101_driver(_so101_robot(uuid4()), _StubFactory())

        recipe = to_config(shared)["init_args"]["robot"]
        assert recipe["class_path"] == "physicalai.robot.SO101"
        assert recipe["init_args"]["port"] == "/dev/ttyACM0"
        assert recipe["init_args"]["role"] == "follower"
        assert recipe["init_args"]["unit"] == "normalized"
        assert set(recipe["init_args"]["calibration"]) == set(JOINT_NAMES)

    async def test_recipe_rebuilds_a_calibrated_driver(self) -> None:
        """The owner process instantiates the recipe; it must yield a usable driver."""
        shared = await _build_so101_driver(_so101_robot(uuid4()), _StubFactory())

        driver = instantiate(to_config(shared)["init_args"]["robot"])
        assert driver.joint_names == list(JOINT_NAMES)

    async def test_leader_role_is_exported(self) -> None:
        robot = _so101_robot(uuid4())
        robot.type = "SO101_Leader"
        shared = await _build_so101_driver(robot, _StubFactory())

        assert to_config(shared)["init_args"]["robot"]["init_args"]["role"] == "leader"

    async def test_missing_calibration_is_rejected(self) -> None:
        robot = _so101_robot(uuid4())
        robot.payload.calibration = None

        with pytest.raises(ValueError, match="calibration is required"):
            await _build_so101_driver(robot, _StubFactory())

    async def test_unresolvable_port_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Could not resolve a serial port"):
            await _build_so101_driver(_so101_robot(uuid4()), _StubFactory(port=None))


class TestTrossenBuilders:
    async def test_single_arm_uses_transport_safe_name(self) -> None:
        robot_id = uuid4()
        robot = TrossenSingleArmRobot(
            id=robot_id,
            name=DISPLAY_NAME,
            type="Trossen_WidowXAI_Follower",
            payload=TrossenSingleArmPayload(connection_string="192.168.1.2"),
        )

        shared = await _build_trossen_single_arm_driver(robot, _StubFactory())

        assert shared.name == str(robot_id)
        recipe = to_config(shared)["init_args"]["robot"]
        assert recipe == {
            "class_path": "physicalai.robot.WidowXAI",
            "init_args": {"ip": "192.168.1.2", "role": "follower"},
        }

    async def test_bimanual_is_owned_as_one_robot(self) -> None:
        robot_id = uuid4()
        robot = TrossenBimanualRobot(
            id=robot_id,
            name=DISPLAY_NAME,
            type="Trossen_Bimanual_WidowXAI_Follower",
            payload=TrossenBimanualPayload(
                connection_string_left="192.168.1.2",
                connection_string_right="192.168.1.3",
            ),
        )

        shared = await _build_trossen_bimanual_driver(robot, _StubFactory())

        assert shared.name == str(robot_id)
        # A single owner holds both arms, rather than one SharedRobot per arm.
        recipe = to_config(shared)["init_args"]["robot"]
        assert recipe["class_path"] == "physicalai.robot.BimanualWidowXAI"
        assert recipe["init_args"]["left"]["init_args"]["ip"] == "192.168.1.2"
        assert recipe["init_args"]["right"]["init_args"]["ip"] == "192.168.1.3"

    async def test_bimanual_recipe_rebuilds_both_arms(self) -> None:
        robot = TrossenBimanualRobot(
            id=uuid4(),
            name=DISPLAY_NAME,
            type="Trossen_Bimanual_WidowXAI_Leader",
            payload=TrossenBimanualPayload(
                connection_string_left="192.168.1.2",
                connection_string_right="192.168.1.3",
            ),
        )

        shared = await _build_trossen_bimanual_driver(robot, _StubFactory())

        driver = instantiate(to_config(shared)["init_args"]["robot"])
        joint_names = driver.joint_names
        assert any(name.startswith("left_") for name in joint_names)
        assert any(name.startswith("right_") for name in joint_names)
