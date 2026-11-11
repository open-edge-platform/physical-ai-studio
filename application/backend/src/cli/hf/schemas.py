# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Structured Hub metadata and results; JSON objects stay at serialization boundaries."""

from collections.abc import Mapping
from dataclasses import field
from typing import Annotated, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.dataclasses import dataclass

type JsonObject = dict[str, JsonValue]
type ResourceKind = Literal["model", "dataset"]

FEATURE_FLAGS = {"lora_enabled", "lora_use_dora", "snapflow_enabled"}
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def to_json(value: "JsonRecord | AssetMetadata | JsonObject") -> JsonObject:
    """Serialize a result or receipt, retaining explicitly supplied null metadata."""
    return cast("JsonObject", TypeAdapter(type(value)).dump_python(value, mode="json", exclude_unset=True))


def feature_flags(properties: Mapping[str, object]) -> dict[str, bool]:
    """Select portable boolean flags without carrying arbitrary training properties."""
    return {key: value for key, value in properties.items() if key in FEATURE_FLAGS and isinstance(value, bool)}


class JsonRecord:
    """Omit unavailable optional fields without dropping nulls from nested JSON."""

    nullable_fields: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> JsonObject:
        return {key: value for key, value in handler(self).items() if value is not None or key in self.nullable_fields}


class AssetMetadata(BaseModel):
    """Portable Studio fields, preserving additional JSON metadata for round trips."""

    model_config = ConfigDict(extra="allow", strict=True, hide_input_in_errors=True)

    name: Annotated[str, Field(min_length=1, max_length=255)] | None = None
    policy: str | None = None
    version: Annotated[int, Field(ge=1)] = 1
    properties: Mapping[str, JsonValue] = Field(default_factory=dict)
    default_task: str = ""
    source_id: str | None = None
    source_project_id: str | None = None
    source_dataset_id: str | None = None
    schema_version: int | None = None
    kind: ResourceKind | None = None
    status: str | None = None
    transfer_id: str | None = None
    files: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Studio asset name must contain 1 to 255 characters")
        return value

    @field_validator("properties")
    @classmethod
    def _boolean_flags(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if any(not isinstance(value[key], bool) for key in FEATURE_FLAGS & value.keys()):
            raise ValueError("Studio model properties must contain boolean feature flags")
        return value


@dataclass
class ModelDetails:
    """Policy and deployment artifacts found during inspection."""

    policy: str | None
    backends: dict[str, JsonObject]
    checkpoints: list[str]


@dataclass
class DatasetDetails:
    """Dataset dimensions and unmodified LeRobot feature descriptions."""

    episodes: int | None
    frames: int | None
    fps: int | float | None
    robot_type: str | None
    features: dict[str, JsonObject]
    cameras: list[str]


class DatasetInfo(BaseModel):
    """LeRobot metadata used by both inspection and payload validation."""

    model_config = ConfigDict(strict=True, hide_input_in_errors=True)

    total_episodes: int | None = None
    total_frames: int | None = None
    fps: int | float | None = None
    robot_type: str | None = None
    codebase_version: str | None = None
    features: dict[str, JsonObject] = Field(default_factory=dict)
    data_path: str = DATA_PATH
    video_path: str = VIDEO_PATH

    @field_validator("features", mode="before")
    @classmethod
    def _feature_objects(cls, value: object) -> object:
        if not isinstance(value, dict) or not all(isinstance(feature, dict) for feature in value.values()):
            raise ValueError("Dataset features must map feature names to objects")
        return value

    @property
    def cameras(self) -> list[str]:
        """Return the features stored as videos."""
        return [name for name, feature in self.features.items() if feature.get("dtype") == "video"]


@dataclass(kw_only=True)
class Inspection(JsonRecord):
    """A metadata-only compatibility report with resource-specific details."""

    nullable_fields: ClassVar[frozenset[str]] = frozenset({"subfolder"})
    kind: ResourceKind
    subfolder: str | None = None
    schema_version: Literal[1] = 1
    validation_level: Literal["metadata"] = "metadata"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    size_bytes: int = 0
    size_complete: bool = True
    studio: AssetMetadata | None = None
    detected_format: Literal["studio_model", "studio_exports", "lerobot_v3"] | None = None
    model: ModelDetails | None = None
    dataset: DatasetDetails | None = None
    required_import_options: list[str] | None = None
    repo_id: str | None = None
    revision: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _consistent_importability(cls, value: object) -> object:
        if isinstance(value, dict) and "importable" in value:
            expected = not value.get("errors", [])
            if value["importable"] is not expected:
                raise ValueError("Saved inspection importability disagrees with its errors")
        return value

    @computed_field  # type: ignore[prop-decorator]  # mypy does not support decorated properties.
    @property
    def importable(self) -> bool:
        """Derive compatibility from errors rather than maintaining a second flag."""
        return not self.errors

    def require_importable(self) -> None:
        """Reject incompatible resources before touching their payloads."""
        if not self.importable:
            raise ValueError("Resource cannot be imported: " + "; ".join(self.errors))

    @property
    def pinned_revision(self) -> str:
        """Require a resolved remote commit before creating a download receipt."""
        if not self.revision:
            raise ValueError("Hub did not return a pinned revision")
        return self.revision


@dataclass(kw_only=True)
class TransferResult(JsonRecord):
    """A completed import or export, independent of its recovery receipt."""

    kind: ResourceKind
    id: str
    repo_id: str
    revision: str
    project_id: str | None = None
    path: str | None = None
    environment_id: str | None = None
    url: str | None = None
