# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validated transfer receipts, preserving the version-one JSON format."""

from typing import Annotated, Any, Literal, NamedTuple, NotRequired, TypedDict

from pydantic import Field, TypeAdapter

type ResourceKind = Literal["model", "dataset"]


class FileFingerprint(NamedTuple):
    """File identity stored as a three-element array in transfer JSON."""

    size: int
    mtime_ns: int
    inode: int


class ExportSource(TypedDict):
    """Managed source directory and portable asset metadata."""

    path: str
    metadata: dict[str, Any]


class BaseState(TypedDict):
    """Fields shared by import and export receipts."""

    schema_version: Literal[1]
    transfer_id: str
    kind: ResourceKind
    repo_id: str
    asset_id: str
    result: NotRequired[dict[str, Any]]


class ImportState(BaseState):
    """Pinned download and local registration options."""

    operation: Literal["import"]
    phase: Literal["prepared", "downloaded", "completed"]
    project_id: str
    revision: str
    inspection: dict[str, Any]
    name: NotRequired[str | None]
    dataset_id: NotRequired[str | None]
    environment_id: NotRequired[str]


class ExportState(BaseState):
    """Source identity and resumable upload options."""

    operation: Literal["export"]
    phase: Literal["prepared", "uploading", "completed"]
    source: ExportSource
    files: list[str]
    fingerprints: dict[str, FileFingerprint]
    public: bool
    update: bool
    url: NotRequired[str]


type TransferState = ImportState | ExportState

STATE_ADAPTER = TypeAdapter(Annotated[TransferState, Field(discriminator="operation")])
