# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest schema and loader for exported model packages.

A manifest describes everything needed to reconstruct an inference
pipeline from a directory of exported artifacts: which runner to
use, what shapes the hardware exposes, etc.

The on-disk format is ``manifest.json``.  Components support two
resolution modes:

1. **type + flat params** — short registry names with flat kwargs,
   written by LeRobot and readable by both frameworks.
2. **class_path + init_args** — fully-qualified class paths with nested
   kwargs (jsonargparse-style), for full-power PhysicalAI usage.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_VERSION = "1.0"
MANIFEST_FORMAT = "policy_package"

# Alias builtin ``type`` so it remains accessible inside classes that
# define a Pydantic field with the same name (e.g. ``ComponentSpec.type``).
_type = type


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


class OrderedTensorSpec(TensorSpec):
    """Tensor spec with named dimension ordering.

    Extends :class:`TensorSpec` with an ``order`` list that records
    the semantic name of each element along the primary axis — e.g.
    joint names for a robot state vector.

    Attributes:
        order: Ordered list of element names (e.g. joint names).
            Defaults to an empty list when ordering is unspecified.
    """

    order: list[str] = Field(default_factory=list)


class RobotSpec(BaseModel):
    """Robot hardware descriptor — state and action spaces.

    Attributes:
        name: Logical name, e.g. ``"main"``.
        type: Robot model string (informational), e.g. ``"Koch v1.1"``.
        state: Expected state tensor spec (with optional joint ordering).
        action: Expected action tensor spec (with optional joint ordering).
    """

    model_config = ConfigDict(frozen=True)
    name: str
    type: str = ""
    state: OrderedTensorSpec | None = None
    action: OrderedTensorSpec | None = None


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


class PolicySource(BaseModel):
    """Origin and training class for a policy.

    Attributes:
        repo_id: HuggingFace or Git repo identifier, e.g.
            ``"lerobot/act_aloha_sim_transfer_cube_human"``.
        class_path: Fully-qualified training class, e.g.
            ``"physicalai.policies.act.policy.ACT"``.  Informational —
            the inference runtime does not import this.
    """

    model_config = ConfigDict(frozen=True)
    repo_id: str = ""
    class_path: str = ""


class PolicySpec(BaseModel):
    """Policy identity section.

    Attributes:
        name: Human-readable policy name, e.g. ``"act"``.
        source: Origin and training class reference.
    """

    model_config = ConfigDict(frozen=True)
    name: str = ""
    source: PolicySource = Field(default_factory=PolicySource)


class ComponentSpec(BaseModel):
    """Dual-resolution component descriptor for dynamic instantiation.

    Supports two resolution modes:

    1. **type + flat params** (LeRobot-compatible)::

        {"type": "action_chunking", "chunk_size": 100, "n_action_steps": 100}

    2. **class_path + init_args** (full-power PhysicalAI)::

        {"class_path": "physicalai.inference.runners.ActionChunking",
         "init_args": {"chunk_size": 100}}

    When ``class_path`` is present it takes precedence.  When only
    ``type`` is present, the :class:`ComponentRegistry` resolves it.

    Attributes:
        type: Registered short name (e.g. ``"action_chunking"``).
        class_path: Fully-qualified class path for direct import.
        init_args: Keyword arguments forwarded to the constructor
            (used with ``class_path`` mode).
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    type: str = ""
    class_path: str = ""
    init_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _must_have_type_or_class_path(self) -> ComponentSpec:
        if not self.type and not self.class_path:
            msg = "ComponentSpec requires either 'type' or 'class_path'"
            raise ValueError(msg)
        return self

    @property
    def flat_params(self) -> dict[str, Any]:
        """Return extra fields as flat params (type-based resolution).

        Returns all fields stored in ``model_extra`` — these are the
        flat kwargs passed alongside ``type`` in LeRobot-style specs.
        """
        return dict(self.model_extra) if self.model_extra else {}

    @classmethod
    def from_class(cls, target: _type, **overrides: Any) -> ComponentSpec:  # noqa: ANN401
        """Build a spec by introspecting a class constructor.

        Parameters not present in *overrides* use their default values.
        Required parameters without defaults must be provided in *overrides*
        or a TypeError is raised.

        For nested components, pass a ``ComponentSpec`` instance (e.g. from
        another ``from_class`` call) — it will be serialized automatically.

        Args:
            target: The class to build a spec for.
            **overrides: Values that override or supply constructor args.

        Returns:
            A ``ComponentSpec`` ready for serialisation or instantiation.

        Raises:
            TypeError: If required parameters are missing from overrides.
        """
        sig = inspect.signature(target)
        init_args: dict[str, Any] = {}
        missing: list[str] = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in overrides:
                value = overrides[name]
            elif param.default is not param.empty:
                value = param.default
            else:
                missing.append(name)
                continue

            if isinstance(value, ComponentSpec):
                value = value.model_dump()
            init_args[name] = value

        if missing:
            msg = (
                f"Missing required parameters for {target.__qualname__}: "
                f"{', '.join(missing)}. Pass them as keyword arguments."
            )
            raise TypeError(msg)

        return cls(
            class_path=f"{target.__module__}.{target.__qualname__}",
            init_args=init_args,
        )


class ModelSpec(BaseModel):
    """Model inference specification.

    Groups the runner, artifacts, preprocessors, and postprocessors
    under a single ``model`` section in the manifest.

    Attributes:
        n_obs_steps: Number of observation steps the model expects.
        runner: Runner component specification.
        artifacts: Mapping of logical name → filename,
            e.g. ``{"model": "model.onnx"}``.
        preprocessors: Pipeline stages applied to observations
            *before* the runner.
        postprocessors: Pipeline stages applied to runner output
            *after* inference.
    """

    model_config = ConfigDict(frozen=True)
    n_obs_steps: int = 1
    runner: ComponentSpec | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    preprocessors: list[ComponentSpec] = Field(default_factory=list)
    postprocessors: list[ComponentSpec] = Field(default_factory=list)


class HardwareSpec(BaseModel):
    """Hardware configuration for robots and cameras.

    Attributes:
        robots: Hardware descriptors for robot state/action.
        cameras: Hardware descriptors for camera inputs.
    """

    model_config = ConfigDict(frozen=True)
    robots: list[RobotSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)


class MetadataSpec(BaseModel):
    """Package metadata.

    Attributes:
        created_at: ISO 8601 timestamp of when the package was created.
        created_by: Tool or framework that created the package.
    """

    model_config = ConfigDict(frozen=True)
    created_at: str = ""
    created_by: str = ""


class Manifest(BaseModel):
    """Parsed manifest for an exported model package.

    Attributes:
        format: Always ``"policy_package"`` for this schema.
        version: Schema version string.
        policy: Policy identity (name, source).
        model: Inference model specification (runner, artifacts,
            preprocessors, postprocessors).
        hardware: Hardware descriptors (robots, cameras).
        metadata: Package metadata (creation info).

    Unknown top-level keys are preserved in ``model_extra`` so that
    domain layers can store additional data without schema changes.
    """

    model_config = ConfigDict(frozen=True, extra="allow")
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    policy: PolicySpec = Field(default_factory=PolicySpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    metadata: MetadataSpec = Field(default_factory=MetadataSpec)

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

        from physicalai.inference.runners.action_chunking import ActionChunking  # noqa: PLC0415
        from physicalai.inference.runners.single_pass import SinglePass  # noqa: PLC0415

        if use_action_queue:
            runner = ComponentSpec.from_class(
                ActionChunking,
                runner=ComponentSpec.from_class(SinglePass),
                chunk_size=chunk_size,
            )
        else:
            runner = ComponentSpec.from_class(SinglePass)

        backend = metadata.get("backend", "")
        artifacts: dict[str, str] = {backend: ""} if backend else {}

        return cls.model_validate({
            "policy": {
                "name": policy_name,
                "source": {"class_path": policy_class},
            },
            "model": {
                "runner": {"class_path": runner.class_path, "init_args": runner.init_args},
                "artifacts": artifacts,
            },
            **{
                k: v
                for k, v in metadata.items()
                if k not in {"policy_class", "backend", "use_action_queue", "chunk_size"}
            },
        })

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
