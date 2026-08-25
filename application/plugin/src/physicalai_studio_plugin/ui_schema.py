"""Typed JSON Schema extensions understood by the Studio robot form."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, Required, TypedDict

if TYPE_CHECKING:
    from pydantic import BaseModel


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


def validate_robot_payload_ui(payload_model: type[BaseModel]) -> None:  # noqa: C901, PLR0915
    """Validate Studio UI metadata emitted by a robot payload model.

    Raises:
        ValueError: If metadata does not conform to the recursive item-list contract.
    """
    schema = payload_model.model_json_schema()
    definitions = schema.get("$defs", {})

    def error(path: str, message: str) -> None:
        error_message = f"{payload_model.__name__} UI schema at {path}: {message}"
        raise ValueError(error_message)

    def resolve(field_schema: dict[str, Any]) -> dict[str, Any]:
        reference = field_schema.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            return field_schema
        name = reference.removeprefix("#/$defs/")
        resolved = definitions.get(name)
        return resolved if isinstance(resolved, dict) else field_schema

    def validate_items(  # noqa: C901, PLR0912
        items: list[object],
        properties: dict[str, Any],
        path: str,
        owned_fields: set[str] | None = None,
    ) -> None:
        owned_fields = set() if owned_fields is None else owned_fields
        for index, item in enumerate(items):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict):
                error(item_path, "must be an object")
                continue
            kind = item.get("kind")

            if kind == "info":
                if not isinstance(item.get("text"), str):
                    error(item_path, "info items require a text string")
                continue

            if kind == "section":
                if not isinstance(item.get("id"), str):
                    error(item_path, "section items require an id string")
                section_items = item.get("items")
                if not isinstance(section_items, list):
                    error(item_path, "section items require a list of items")
                    continue
                validate_items(section_items, properties, f"{item_path}.items", owned_fields)
                continue

            if kind == "field":
                name = item.get("name")
                if not isinstance(name, str) or name not in properties:
                    error(item_path, "field items must reference an existing payload field")
                    continue
                if name in owned_fields:
                    error(item_path, f"field '{name}' is owned more than once")
                owned_fields.add(name)
                continue

            if kind == "connection":
                bindings = item.get("bind")
                if not isinstance(bindings, dict) or not isinstance(bindings.get("connection"), str):
                    error(item_path, "connection items require bind.connection")
                    continue
                for binding_name in ("connection", "serial_number"):
                    field_name = bindings.get(binding_name)
                    if field_name is None:
                        continue
                    if not isinstance(field_name, str) or field_name not in properties:
                        error(item_path, f"bind.{binding_name} must reference an existing payload field")
                    if resolve(properties[field_name]).get("type") != "string":
                        error(item_path, f"bind.{binding_name} must reference a string payload field")
                    if field_name in owned_fields:
                        error(item_path, f"field '{field_name}' is owned more than once")
                    owned_fields.add(field_name)
                continue

            error(item_path, "has an unsupported or missing kind")

    visited: set[int] = set()

    def validate_model_schema(model_schema: object, path: str) -> None:
        if not isinstance(model_schema, dict) or id(model_schema) in visited:
            return
        visited.add(id(model_schema))
        properties = model_schema.get("properties")
        ui = model_schema.get("x-physicalai-ui")
        if ui is not None:
            if not isinstance(ui, list):
                error(f"{path}.x-physicalai-ui", "must be a list of items")
            else:
                validate_items(ui, properties if isinstance(properties, dict) else {}, f"{path}.x-physicalai-ui")
        if isinstance(properties, dict):
            for field_name, field_schema in properties.items():
                resolved = resolve(field_schema) if isinstance(field_schema, dict) else field_schema
                if isinstance(resolved, dict) and isinstance(resolved.get("properties"), dict):
                    validate_model_schema(resolved, f"{path}.properties.{field_name}")

    validate_model_schema(schema, "$")
    for name, definition in definitions.items():
        validate_model_schema(definition, f"$.$defs.{name}")
