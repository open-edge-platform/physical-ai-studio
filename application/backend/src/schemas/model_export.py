# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Portable export metadata and package-local filesystem access."""

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Self
from uuid import UUID

from pydantic import BaseModel

from utils.filesystem import FileTree

BACKENDS = frozenset({"openvino", "onnx", "torch", "executorch"})


class ExportInfo(BaseModel):
    created_at: datetime | None = None
    job_id: UUID | None = None


@dataclass(frozen=True)
class ExportPackage:
    """A lazily read package snapshot; create a new instance after changing its files."""

    directory: Path
    backend: str

    @classmethod
    def for_model(cls, model_path: Path, backend: str) -> Self:
        """Resolve a package location, rejecting unknown names and symbolic links."""
        if backend not in BACKENDS:
            raise ValueError("Unknown export backend")
        return cls(FileTree(model_path.absolute()).resolve(Path("exports") / backend), backend)

    @cached_property
    def files(self) -> tuple[Path, ...]:
        """Return package files without following symbolic links."""
        return tuple(FileTree(self.directory).files())

    @cached_property
    def manifest(self) -> dict:
        """Read the manifest and validate its local artifact files."""
        path = FileTree(self.directory).resolve(Path("manifest.json"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifacts = manifest["model"]["artifacts"]
        if not isinstance(artifacts, dict) or self.backend not in artifacts:
            raise ValueError("Missing backend artifact")
        files = {path.relative_to(self.directory).as_posix() for path in self.files}

        def require_file(value: str) -> None:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() not in files:
                raise ValueError(f"Missing or unsafe export artifact: {value}")
            if relative.suffix == ".xml" and relative.with_suffix(".bin").as_posix() not in files:
                raise ValueError(f"Missing weights for export artifact: {value}")

        for artifact in artifacts.values():
            if not isinstance(artifact, str):
                raise ValueError("Artifact paths must be strings")
            require_file(artifact)

        def validate_components(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "artifact" and isinstance(item, str):
                        require_file(item)
                    else:
                        validate_components(item)
            elif isinstance(value, list):
                for item in value:
                    validate_components(item)

        validate_components(manifest["model"])
        return manifest

    @property
    def is_complete(self) -> bool:
        """Whether the package has a valid manifest and all required artifacts."""
        try:
            self.manifest
        except (KeyError, TypeError, ValueError, OSError):
            return False
        return True

    @property
    def artifact(self) -> Path:
        return self.directory / self.manifest["model"]["artifacts"][self.backend]

    @property
    def info(self) -> ExportInfo:
        """Read metadata without requiring it on existing exports."""
        return ExportInfo.model_validate(self.manifest.get("studio", {}).get("export", {}))
