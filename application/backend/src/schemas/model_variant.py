# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Portable export metadata and package-local filesystem access."""

import json
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel

from utils.filesystem import FileTree

VariantId = Literal["original", "int8"]
VARIANT_IDS: frozenset[VariantId] = frozenset({"original", "int8"})
BACKENDS = frozenset({"openvino", "onnx", "torch", "executorch"})


class VariantInfo(BaseModel):
    schema_version: Literal[1] = 1
    id: VariantId
    weight_compression: Literal["int8_sym"] | None = None
    created_at: datetime | None = None
    job_id: UUID | None = None


@dataclass(frozen=True)
class ExportPackage:
    """A lazily read package snapshot; create a new instance after changing its files."""

    directory: Path
    backend: str
    variant: VariantId = "original"

    @classmethod
    def for_model(cls, model_path: Path, backend: str, variant: VariantId = "original") -> Self:
        """Resolve a package location, rejecting unknown names and symbolic links."""
        if backend not in BACKENDS or variant not in VARIANT_IDS:
            raise ValueError("Unknown export backend or variant")
        relative = Path("exports") / backend
        if variant != "original":
            relative = relative / "variants" / variant
        return cls(FileTree(model_path.absolute()).resolve(relative), backend, variant)

    @cached_property
    def files(self) -> tuple[Path, ...]:
        """Package files, excluding sibling variants."""
        return tuple(FileTree(self.directory).files(exclude=("variants",)))

    @cached_property
    def manifest(self) -> dict:
        """The manifest, validated against the package's local artifact files."""
        path = FileTree(self.directory).resolve(Path("manifest.json"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        artifacts = manifest["model"]["artifacts"]
        if not isinstance(artifacts, dict) or self.backend not in artifacts:
            raise ValueError("Missing backend artifact")
        if self.backend == "openvino" and (
            not isinstance(artifacts[self.backend], str) or Path(artifacts[self.backend]).suffix != ".xml"
        ):
            raise ValueError("OpenVINO policy artifact must be XML")
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
    def artifact(self) -> Path:
        return self.directory / self.manifest["model"]["artifacts"][self.backend]

    @property
    def info(self) -> VariantInfo:
        """Read metadata without requiring it on existing original exports."""
        studio = self.manifest.get("studio", {})
        if not isinstance(studio, dict):
            raise ValueError("Manifest studio metadata must be an object")
        metadata = studio.get("variant")
        if metadata is not None:
            info = VariantInfo.model_validate(metadata)
            return info.model_copy(update={"id": self.variant})
        return VariantInfo(
            id=self.variant,
            weight_compression="int8_sym" if self.variant == "int8" else None,
        )
