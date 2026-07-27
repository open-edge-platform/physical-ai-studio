from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from physicalai_studio_plugin import (
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
)


class TestPayload(BaseModel):
    serial_number: str = Field(...)
    connection_string: str = ""


class TestProbe:
    """Structurally implements RobotProbe[TestPayload]."""

    async def discover(self, manager: PortScanner) -> list[SerialPortInfo]:
        return []

    async def identify(
        self,
        payload: TestPayload,
        manager: PortScanner | None,
        joint: str | None = None,
    ) -> None:
        self._last_payload = payload

    async def is_online(
        self,
        payload: TestPayload,
        manager: PortScanner | None = None,
    ) -> bool:
        return payload.serial_number != ""


def test_probe_is_runtime_checkable() -> None:
    probe = TestProbe()
    assert isinstance(probe, RobotProbe)


def test_typed_payload_reaches_identify() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="SN-001")

    asyncio.run(probe.identify(payload, None))
    assert probe._last_payload is payload


def test_is_online_typed_payload() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="SN-001")

    result = asyncio.run(probe.is_online(payload))
    assert result


def test_is_online_empty_payload() -> None:
    probe = TestProbe()
    payload = TestPayload(serial_number="")

    result = asyncio.run(probe.is_online(payload))
    assert not result


def test_definition_creation() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    probe = TestProbe()
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
        adapter_options=RobotAdapterOptions(include_velocities=True),
        probe=probe,
    )

    assert definition.type == "Test_Follower"
    assert definition.robot_payload is TestPayload
    assert definition.probe is probe


def test_generic_payload_linked_to_probe() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    probe = TestProbe()
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
        probe=probe,
    )

    payload_instance = TestPayload(serial_number="SN-002")
    assert definition.probe is not None
    asyncio.run(definition.probe.identify(payload_instance, None))
    assert probe._last_payload is payload_instance


def test_multiple_definitions() -> None:
    definitions: list[RobotCatalogDefinition] = [
        RobotCatalogDefinition[TestPayload](
            type="RobotA",
            display_name="Robot A",
            role="follower",
            robot_payload=TestPayload,
            probe=TestProbe(),
        ),
        RobotCatalogDefinition[TestPayload](
            type="RobotB",
            display_name="Robot B",
            role="leader",
            robot_payload=TestPayload,
            probe=TestProbe(),
        ),
    ]
    assert len(definitions) == 2
    assert definitions[0].type == "RobotA"
    assert definitions[1].role == "leader"


def test_valid_payload_passes_validation() -> None:
    asset = RobotAsset(
        urdf_relative_path=Path("test/model.urdf"),
        packages={"test": Path("test")},
        joint_map={"gripper.pos": ["gripper"]},
    )
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
        asset=asset,
    )

    raw = {"serial_number": "SN-003", "connection_string": "/dev/ttyUSB0"}
    payload_model = definition.robot_payload
    assert payload_model is not None
    validated = payload_model.model_validate(raw)
    assert isinstance(validated, BaseModel)
    assert validated.serial_number == "SN-003"


def test_invalid_payload_raises() -> None:
    definition = RobotCatalogDefinition[TestPayload](
        type="Test_Follower",
        display_name="Test Follower",
        role="follower",
        robot_payload=TestPayload,
    )

    raw = {"connection_string": "/dev/ttyUSB0"}
    payload_model = definition.robot_payload
    assert payload_model is not None
    with pytest.raises(ValidationError):
        payload_model.model_validate(raw)


def test_no_payload_model_returns_raw_dict() -> None:
    definition = RobotCatalogDefinition[TestPayload](
        type="NoPayload",
        display_name="No Payload",
        role="follower",
        robot_payload=None,
    )

    raw = {"some": "data"}
    result = raw if definition.robot_payload is None else definition.robot_payload.model_validate(raw)
    assert result == raw
