from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physicalai_studio_plugin import (
    PortScanner,
    RobotAdapterOptions,
    RobotAsset,
    RobotCatalogDefinition,
    RobotProbe,
    SerialPortInfo,
    robot_field_ui,
    robot_payload_ui,
    validate_robot_payload_ui,
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


def test_robot_field_ui_supports_required_option() -> None:
    assert robot_field_ui({"required": True}) == {"x-physicalai-ui": {"required": True}}


def test_robot_payload_ui_supports_recursive_items() -> None:
    assert robot_payload_ui(
        [
            {
                "kind": "section",
                "id": "connection",
                "title": "Connection",
                "description": "Pick a detected device or enter one manually.",
                "items": [
                    {"kind": "info", "text": "USB hubs can rename ports after reboot.", "variant": "warning"},
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            },
        ],
    ) == {
        "x-physicalai-ui": [
            {
                "kind": "section",
                "id": "connection",
                "title": "Connection",
                "description": "Pick a detected device or enter one manually.",
                "items": [
                    {"kind": "info", "text": "USB hubs can rename ports after reboot.", "variant": "warning"},
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            },
        ],
    }


def test_validate_robot_payload_ui_accepts_nested_item_lists() -> None:
    class ConnectionPayload(BaseModel):
        connection_string: str
        serial_number: str

        model_config = ConfigDict(
            json_schema_extra=robot_payload_ui(
                [
                    {
                        "kind": "connection",
                        "bind": {"connection": "connection_string", "serial_number": "serial_number"},
                    },
                ],
            ),
        )

    class RobotPayload(BaseModel):
        arm: ConnectionPayload

        model_config = ConfigDict(json_schema_extra=robot_payload_ui([{"kind": "field", "name": "arm"}]))

    validate_robot_payload_ui(RobotPayload)


def test_validate_robot_payload_ui_ignores_field_options() -> None:
    class Payload(BaseModel):
        id: str | None = Field(default=None, json_schema_extra=robot_field_ui({"required": True}))

    validate_robot_payload_ui(Payload)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ({"groups": {}}, "must be a list of items"),
        ([{"kind": "field", "name": "missing"}], "must reference an existing payload field"),
        ([{"kind": "connection", "bind": {"connection": "port"}}], "must reference a string payload field"),
        (
            [
                {"kind": "field", "name": "connection_string"},
                {"kind": "connection", "bind": {"connection": "connection_string"}},
            ],
            "owned more than once",
        ),
    ],
)
def test_validate_robot_payload_ui_rejects_invalid_metadata(items: object, message: str) -> None:
    class InvalidPayload(BaseModel):
        connection_string: str
        port: int

        model_config = ConfigDict(json_schema_extra={"x-physicalai-ui": items})

    with pytest.raises(ValueError, match=message):
        validate_robot_payload_ui(InvalidPayload)
