"""Typed JSON Schema extensions understood by the Studio robot form."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict


class RobotUiInfoItem(TypedDict, total=False):
    """Read-only informational text shown in the Studio robot form."""

    kind: Required[Literal["info"]]
    title: NotRequired[str]
    text: Required[str]
    variant: NotRequired[Literal["info", "warning"]]


class RobotFieldUiOptions(TypedDict, total=False):
    """Per-field UI overrides understood by the Studio robot form."""

    required: bool


class RobotUiConnectionBinding(TypedDict, total=False):
    """Payload field bindings for the connection control."""

    connection: Required[str]
    serial_number: NotRequired[str]


class RobotUiConnectionItem(TypedDict, total=False):
    """Options for the first-party connection control."""

    kind: Required[Literal["connection"]]
    label: NotRequired[str]
    description: NotRequired[str]
    device_discovery: NotRequired[bool]
    identify: NotRequired[bool]
    manual_entry: NotRequired[bool]
    bind: Required[RobotUiConnectionBinding]


class RobotUiFieldItem(TypedDict):
    """A standard payload field rendered in the form."""

    kind: Required[Literal["field"]]
    name: Required[str]


class RobotUiSectionOptions(TypedDict, total=False):
    """A recursively rendered section of form items."""

    kind: Required[Literal["section"]]
    id: Required[str]
    title: NotRequired[str]
    description: NotRequired[str]
    items: Required[list[RobotUiItem]]


RobotUiItem = RobotUiInfoItem | RobotUiConnectionItem | RobotUiFieldItem | RobotUiSectionOptions

RobotPayloadUiOptions = list[RobotUiItem]


FieldSchemaExtra = TypedDict("FieldSchemaExtra", {"x-physicalai-ui": RobotFieldUiOptions})
ModelSchemaExtra = TypedDict("ModelSchemaExtra", {"x-physicalai-ui": RobotPayloadUiOptions})


def robot_field_ui(options: RobotFieldUiOptions) -> FieldSchemaExtra:
    """Create typed ``Field(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": options}


def robot_payload_ui(items: RobotPayloadUiOptions) -> ModelSchemaExtra:
    """Create typed ``ConfigDict(json_schema_extra=...)`` metadata."""
    return {"x-physicalai-ui": items}
