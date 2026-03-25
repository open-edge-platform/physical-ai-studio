# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest schema and loader for exported model packages.

A manifest describes everything needed to reconstruct an inference
pipeline from a directory of exported artifacts: which runner to
use, which adapter, what shapes the hardware exposes, etc.

The on-disk format is ``manifest.json``.  The schema follows the
``class_path`` + ``init_args`` convention (jsonargparse-style) so that
domain layers can specify their own components without inferencekit
needing to know about them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MANIFEST_VERSION = "1.0"
MANIFEST_FORMAT = "policy_package"


class TensorSpec(BaseModel):
    """Shape and dtype descriptor for one tensor.

    Attributes:
        shape: Tensor shape (no batch dimension).
        dtype: Numpy-compatible dtype string, e.g. ``"float32"``, ``"uint8"``.
    """

    model_config = ConfigDict(frozen=True)
    shape: list[int]
    dtype: str = "float32"

    @field_validator("shape")
    @classmethod
    def _shape_must_be_positive(cls, v: list[int]) -> list[int]:
        if not all(dim > 0 for dim in v):
            msg = "All shape dimensions must be positive integers"
            raise ValueError(msg)
        return v

    @field_validator("dtype")
    @classmethod
    def _dtype_must_be_nonempty(cls, v: str) -> str:
        if not v:
            msg = "dtype must be a non-empty string"
            raise ValueError(msg)
        return v


class RobotSpec(BaseModel):
    """Robot hardware descriptor — state and action spaces.

    Attributes:
        name: Logical name, e.g. ``"main"``.
        type: Robot model string (informational), e.g. ``"Koch v1.1"``.
        state: Expected state tensor spec.
        action: Expected action tensor spec.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    type: str = ""
    state: TensorSpec | None = None
    action: TensorSpec | None = None


class CameraSpec(BaseModel):
    """Camera descriptor — image shape and dtype.

    Attributes:
        name: Logical name, e.g. ``"top"``, ``"wrist"``.
        shape: ``(C, H, W)`` tensor shape.
        dtype: Numpy-compatible dtype string.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    shape: list[int] = Field(default_factory=list)
    dtype: str = "uint8"

    @field_validator("shape")
    @classmethod
    def _shape_must_have_three_elements(cls, v: list[int]) -> list[int]:
        expected_dims = 3
        if v and len(v) != expected_dims:
            msg = f"CameraSpec shape must have exactly 3 elements (C, H, W), got {len(v)}"
            raise ValueError(msg)
        return v


class PolicySpec(BaseModel):
    """Policy identity section.

    Attributes:
        name: Human-readable policy name, e.g. ``"act"``.
        kind: Runner kind hint.  Built-in models use ``"single_pass"``
            or ``"action_chunking"``.  Custom models specify a ``runner``
            section instead.
        class_path: Fully-qualified training class, e.g.
            ``"physicalai.policies.act.ACT"``.  Informational — the
            inference runtime does not import this.
    """

    model_config = ConfigDict(frozen=True)
    name: str = ""
    kind: str = "single_pass"
    class_path: str = ""


class ComponentSpec(BaseModel):
    """A ``class_path`` + ``init_args`` pair for dynamic instantiation.

    Used for runners, adapters, preprocessors, and postprocessors.

    Attributes:
        class_path: Fully-qualified class path, e.g.
            ``"physicalai.inference.runners.SinglePass"``, or a
            registered short name like ``"single_pass"``.
        init_args: Keyword arguments forwarded to the class constructor.
    """

    model_config = ConfigDict(frozen=True)
    class_path: str
    init_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _class_path_must_be_nonempty(cls, v: str) -> str:
        if not v:
            msg = "class_path must be a non-empty string"
            raise ValueError(msg)
        return v


class Manifest(BaseModel):
    """Parsed manifest for an exported model package.

    Attributes:
        format: Always ``"policy_package"`` for this schema.
        version: Schema version string.
        policy: Policy identity (name, kind, training class).
        artifacts: Mapping of backend name → filename,
            e.g. ``{"onnx": "model.onnx"}``.
        runner: Optional dynamic runner specification.
        adapter: Optional dynamic adapter specification.
        robots: Hardware descriptors for robot state/action.
        cameras: Hardware descriptors for camera inputs.

    Unknown top-level keys are preserved in ``model_extra`` so that
    domain layers can store additional data without schema changes.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    policy: PolicySpec = Field(default_factory=PolicySpec)
    artifacts: dict[str, str] = Field(default_factory=dict)
    runner: ComponentSpec | None = None
    adapter: ComponentSpec | None = None
    robots: list[RobotSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load a manifest from a JSON file or directory.

        If *path* is a directory, looks for ``manifest.json`` inside it.

        Args:
            path: Path to ``manifest.json`` or a directory containing it.

        Returns:
            Parsed ``Manifest`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if p.is_dir():
            p /= "manifest.json"
        if not p.exists():
            msg = f"Manifest not found: {p}"
            raise FileNotFoundError(msg)
        with p.open(encoding="utf-8") as fh:
            return cls.model_validate(json.load(fh))

    @classmethod
    def from_legacy_metadata(cls, metadata: dict[str, Any]) -> Manifest:
        """Upgrade a legacy ``metadata.yaml`` dict to a ``Manifest``.

        Maps flat keys (``policy_class``, ``backend``, ``use_action_queue``,
        ``chunk_size``) into the structured manifest schema so that
        downstream code can use a single Manifest type regardless of the
        on-disk format.

        Args:
            metadata: Dict loaded from ``metadata.yaml`` or ``metadata.json``.

        Returns:
            A ``Manifest`` populated with as much information as the legacy
            format provides.
        """
        policy_class = metadata.get("policy_class", "")
        policy_name = _policy_name_from_class_path(policy_class)

        use_action_queue = metadata.get("use_action_queue", False)
        chunk_size = metadata.get("chunk_size", 1)
        kind = "action_chunking" if use_action_queue else "single_pass"
        runner = _build_runner_spec(kind, chunk_size)

        backend = metadata.get("backend", "")
        artifacts: dict[str, str] = {backend: ""} if backend else {}

        data: dict[str, Any] = {
            "policy": {"name": policy_name, "kind": kind, "class_path": policy_class},
            "artifacts": artifacts,
            "runner": {"class_path": runner.class_path, "init_args": runner.init_args},
            "robots": [],
            "cameras": [],
            **metadata,
        }
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        """Write the manifest as ``manifest.json``.

        Args:
            path: File path (typically ``export_dir / "manifest.json"``).
        """
        data = self.model_dump(exclude_defaults=True)
        data.setdefault("format", self.format)
        data.setdefault("version", self.version)
        if "policy" not in data:
            data["policy"] = self.policy.model_dump()
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")


from physicalai.inference.component_factory import component_registry  # noqa: E402


def _build_runner_spec(kind: str, chunk_size: int = 1) -> ComponentSpec:
    """Build a ``ComponentSpec`` for a built-in runner kind.

    Args:
        kind: One of ``"single_pass"`` or ``"action_chunking"``.
        chunk_size: Chunk size for action-chunking runners.

    Returns:
        A ``ComponentSpec`` that can be instantiated later.
    """
    class_path = component_registry.resolve(kind)
    if class_path == kind:
        class_path = component_registry.resolve("single_pass")

    if kind == "action_chunking":
        return ComponentSpec(
            class_path=class_path,
            init_args={
                "runner": {
                    "class_path": component_registry.resolve("single_pass"),
                    "init_args": {},
                },
                "chunk_size": chunk_size,
            },
        )
    return ComponentSpec(class_path=class_path, init_args={})


def _policy_name_from_class_path(class_path: str) -> str:
    """Extract a short policy name from a fully-qualified class path.

    ``"physicalai.policies.act.policy.ACT"`` → ``"act"``

    Args:
        class_path: Dotted class path string.

    Returns:
        Short lowercase name, or empty string if extraction fails.
    """
    parts = class_path.lower().split(".")
    min_parts = 3
    if len(parts) >= min_parts:
        return parts[-2]
    return ""
