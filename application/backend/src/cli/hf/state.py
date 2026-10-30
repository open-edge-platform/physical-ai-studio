# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validated transfer receipts, preserving the version-one JSON format."""

import json
import os
import tempfile
from dataclasses import field
from pathlib import Path
from typing import Literal, NamedTuple, Self
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from cli.hf.schemas import AssetMetadata, Inspection, JsonObject, JsonRecord, ResourceKind, TransferResult, to_json


class FileFingerprint(NamedTuple):
    """File identity stored as a three-element array in transfer JSON."""

    size: int
    mtime_ns: int
    inode: int


@dataclass(frozen=True)
class ExportSource:
    """Managed source directory and portable asset metadata."""

    path: Path
    metadata: AssetMetadata


@dataclass(kw_only=True)
class TransferState[Operation: str, Phase: str](JsonRecord):
    """A validated receipt with its managed storage and recovery operations."""

    kind: ResourceKind
    repo_id: str
    asset_id: str
    operation: Operation
    phase: Phase
    schema_version: Literal[1] = 1
    transfer_id: str = field(default_factory=lambda: str(uuid4()))
    result: TransferResult | None = None

    @property
    def directory(self) -> Path:
        """Return the managed directory for this transfer."""
        from settings import get_settings

        return get_settings().cache_dir / "hf/transfers" / self.transfer_id

    @property
    def resume_command(self) -> str:
        """Return the CLI command that resumes this transfer."""
        return f"physicalai-studio hf {self.kind} {self.operation} --resume {self.transfer_id}"

    @property
    def receipt(self) -> JsonObject:
        """Add recovery details to a completed result."""
        if self.result is None:
            raise ValueError("Transfer has not completed")
        return {**to_json(self.result), "transfer_id": self.transfer_id, "resume_command": self.resume_command}

    def save(self) -> None:
        """Persist state atomically, retaining version-one fingerprint arrays."""
        atomic_json(self.directory / "transfer.json", to_json(self))

    @classmethod
    def load(cls, resume: UUID, kind: ResourceKind) -> Self:
        """Validate a saved receipt before resuming the requested operation."""
        from settings import get_settings

        path = get_settings().cache_dir / "hf/transfers" / str(resume) / "transfer.json"
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not {"schema_version", "transfer_id", "operation", "phase"} <= data.keys():
                raise ValueError("Incomplete transfer receipt")
            state = TypeAdapter(cls).validate_python(data)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read transfer {resume}. Use the original STORAGE_DIR.") from exc
        if state.kind != kind or state.transfer_id != str(resume):
            raise ValueError("Transfer resource kind or ID does not match this command")
        return state


@dataclass(kw_only=True)
class ImportState(TransferState[Literal["import"], Literal["prepared", "downloaded", "completed"]]):
    """Pinned download and local registration options."""

    operation: Literal["import"] = "import"
    phase: Literal["prepared", "downloaded", "completed"] = "prepared"
    project_id: str
    revision: str
    inspection: Inspection
    name: str | None = None
    dataset_id: str | None = None
    environment_id: str | None = None


@dataclass(kw_only=True)
class ExportState(TransferState[Literal["export"], Literal["prepared", "uploading", "completed"]]):
    """Source identity and resumable upload options."""

    operation: Literal["export"] = "export"
    phase: Literal["prepared", "uploading", "completed"] = "prepared"
    source: ExportSource
    files: list[str]
    fingerprints: dict[str, FileFingerprint]
    public: bool
    update: bool

    def preview(self, size_bytes: int) -> JsonObject:
        """Describe a planned export without persisting or uploading it."""
        return {
            "dry_run": True,
            "kind": self.kind,
            "repo_id": self.repo_id,
            "files": [*self.files],
            "size_bytes": size_bytes,
            "private_if_created": not self.public,
        }

    @property
    def url(self) -> str:
        """Return the Hub URL without persisting derivable state."""
        prefix = "datasets/" if self.kind == "dataset" else ""
        return f"https://huggingface.co/{prefix}{self.repo_id}"


def atomic_json(path: Path, value: JsonObject) -> None:
    """Replace a JSON document atomically on its current filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
