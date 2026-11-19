# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write and read Physical AI Runtime packages for GR00T checkpoints.

Layout::

    <package>/
      manifest.json                 Runtime manifest (backend ``gr00t``)
      gr00t_n1d7.gr00t/
        package.json                format, embodiment, I/O mapping, provenance
        model/                      native checkpoint (format ``native``)
        openvino/                   Robotics AI Suite single IR (format ``openvino``)

Packages contain model data and provenance only. Executable code stays in the
installed plugin and its pinned worker environments.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

POLICY = "gr00t_n1d7"
BACKEND = "gr00t"
ARTIFACT_SUFFIX = ".gr00t"
ARTIFACT_NAME = f"{POLICY}{ARTIFACT_SUFFIX}"
DESCRIPTOR_NAME = "package.json"
RUN_INFO_NAME = "physicalai_run.json"
FORMAT_NATIVE = "native"
FORMAT_OPENVINO = "openvino"
_FORMAT_VERSION = 1
_RESET_CALLBACK = "physicalai_gr00t_policy_plugin.runtime.Gr00tResetCallback"
_NATIVE_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "model.safetensors",
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
    "processor/*.json",
    "experiment_cfg/*.json",
    "experiment_cfg/*.yaml",
    RUN_INFO_NAME,
)
_PROCESSOR_FILES = ("processor_config.json", "statistics.json", "embodiment_id.json")


@dataclass(frozen=True)
class IoSpec:
    """Map Runtime observations and actions to GR00T modality keys.

    Attributes:
        cameras: GR00T video key to Runtime camera name.
        state: GR00T state key to its ``[start, end)`` slice of the Runtime state vector.
        action: GR00T action key to its ``[start, end)`` slice of the flat Runtime action.
    """

    cameras: dict[str, str]
    state: dict[str, tuple[int, int]]
    action: dict[str, tuple[int, int]]

    @property
    def state_dim(self) -> int:
        """Length of the Runtime state vector."""
        return max(end for _, end in self.state.values())

    @property
    def action_dim(self) -> int:
        """Length of one flat Runtime action."""
        return max(end for _, end in self.action.values())

    @classmethod
    def from_modality(cls, path: Path) -> IoSpec:
        """Derive the mapping from a LeRobot v2.1 ``meta/modality.json``.

        Returns:
            The I/O mapping recorded in the dataset.
        """
        modality = json.loads(path.read_text(encoding="utf-8"))
        cameras = {
            key: str(spec.get("original_key", f"observation.images.{key}")).removeprefix("observation.images.")
            for key, spec in modality["video"].items()
        }
        return cls(cameras=cameras, state=_slices(modality["state"]), action=_slices(modality["action"]))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping.

        Returns:
            Mapping with ``cameras``, ``state``, and ``action`` entries.
        """
        return {
            "cameras": dict(self.cameras),
            "state": {key: list(bounds) for key, bounds in self.state.items()},
            "action": {key: list(bounds) for key, bounds in self.action.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IoSpec:
        """Rebuild a mapping written by :meth:`to_dict`.

        Returns:
            The parsed mapping.
        """
        return cls(
            cameras={str(key): str(camera) for key, camera in value["cameras"].items()},
            state=_slices(value["state"]),
            action=_slices(value["action"]),
        )


def read_run_info(checkpoint: Path) -> dict[str, Any]:
    """Read the plugin's training record next to a native checkpoint.

    Returns:
        The training record, or an empty mapping for checkpoints trained elsewhere.
    """
    path = checkpoint / RUN_INFO_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_native_package(
    checkpoint: Path,
    output: Path,
    *,
    io: IoSpec,
    run_info: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> Path:
    """Package a native checkpoint for the plugin's PyTorch Runtime path.

    Args:
        checkpoint: Native GR00T checkpoint.
        output: New package directory.
        io: Runtime I/O mapping.
        run_info: Training record of the checkpoint.
        environment: Provenance of the environment that runs the package.

    Returns:
        The completed package directory.
    """

    def populate(artifact: Path) -> dict[str, Any]:
        _copy_allowlisted(checkpoint, artifact / "model", _NATIVE_FILES)
        return {"format": FORMAT_NATIVE, "model_dir": "model", "environment": dict(environment)}

    return _write_package(output, io=io, run_info=run_info, populate=populate)


def write_openvino_package(
    export: Callable[[Path], dict[str, Any]],
    output: Path,
    *,
    checkpoint: Path,
    io: IoSpec,
    run_info: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> Path:
    """Package an OpenVINO export produced by *export* into ``openvino/``.

    Args:
        export: Callable that writes the Robotics AI Suite single IR into the
            directory it receives and returns the exporter's record.
        output: New package directory.
        checkpoint: Native checkpoint that was exported.
        io: Runtime I/O mapping.
        run_info: Training record of the checkpoint.
        environment: Provenance of the patched environment that exported and runs the package.

    Returns:
        The completed package directory.
    """

    def populate(artifact: Path) -> dict[str, Any]:
        ov_dir = artifact / "openvino"
        result = export(ov_dir)
        _complete_ov_config(checkpoint, ov_dir / "config")
        xml = sorted(path.name for path in ov_dir.glob("*.xml"))
        return {
            "format": FORMAT_OPENVINO,
            "model_dir": "openvino",
            "environment": dict(environment),
            "export": {
                "ir": xml,
                "precision": result.get("precision"),
                "verification": result.get("verification"),
                "llm_lang_tokens": result.get("llm_lang_tokens"),
            },
        }

    return _write_package(output, io=io, run_info=run_info, populate=populate)


def load_descriptor(artifact: Path) -> dict[str, Any]:
    """Read and validate ``package.json`` from a Runtime artifact directory.

    Returns:
        The package descriptor.

    Raises:
        FileNotFoundError: If the descriptor is missing.
        ValueError: If the descriptor is invalid.
    """
    path = artifact / DESCRIPTOR_NAME
    if not path.is_file():
        msg = f"GR00T package descriptor not found: {path}"
        raise FileNotFoundError(msg)
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if descriptor.get("format_version") != _FORMAT_VERSION or descriptor.get("policy") != POLICY:
        msg = f"Unsupported GR00T package descriptor in {path}."
        raise ValueError(msg)
    if descriptor.get("format") not in {FORMAT_NATIVE, FORMAT_OPENVINO}:
        msg = f"Unknown GR00T package format {descriptor.get('format')!r}."
        raise ValueError(msg)
    model_dir = (artifact / descriptor["model_dir"]).resolve()
    if not model_dir.is_relative_to(artifact.resolve()) or not model_dir.is_dir():
        msg = "GR00T package model directory is missing or escapes the artifact."
        raise ValueError(msg)
    environment = descriptor.get("environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("name"), str):
        msg = "GR00T package descriptor does not record its worker environment."
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]  # Invalid package content, not a caller type error.
    return descriptor


def _write_package(
    output: Path,
    *,
    io: IoSpec,
    run_info: Mapping[str, Any],
    populate: Callable[[Path], dict[str, Any]],
) -> Path:
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        msg = f"Export directory already exists: {destination}"
        raise FileExistsError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staged = Path(temporary) / "package"
        artifact = staged / ARTIFACT_NAME
        artifact.mkdir(parents=True)
        details = populate(artifact)
        descriptor = {
            "format_version": _FORMAT_VERSION,
            "policy": POLICY,
            "embodiment_tag": run_info.get("embodiment_tag"),
            "io": io.to_dict(),
            "backbone_source": run_info.get("backbone_source"),
            "backbone_override": bool(run_info.get("backbone_override")),
            "training": dict(run_info),
            **details,
        }
        (artifact / DESCRIPTOR_NAME).write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
        _write_manifest(staged)
        staged.rename(destination)
    return destination


def _write_manifest(package: Path) -> None:
    from physicalai.inference.manifest import (  # ruff: ignore[import-outside-top-level]  # Runtime import only when packaging.
        ComponentSpec,
        Manifest,
        MetadataSpec,
        ModelSpec,
        PolicySource,
        PolicySpec,
    )

    manifest = Manifest(
        policy=PolicySpec(name=POLICY, source=PolicySource()),
        model=ModelSpec(
            artifacts={BACKEND: ARTIFACT_NAME},
            callbacks=[ComponentSpec(class_path=_RESET_CALLBACK, init_args={})],
        ),
        metadata=MetadataSpec(created_by="physicalai-gr00t-policy-plugin"),
    )
    manifest.save(package / "manifest.json")


def _copy_allowlisted(source: Path, destination: Path, patterns: Iterable[str]) -> None:
    root = source.resolve(strict=True)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
            continue
        if path.is_symlink():
            msg = f"Checkpoint file {relative} is a symbolic link; materialize the checkpoint first."
            raise ValueError(msg)
        if path.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if not (destination / "config.json").is_file():
        msg = f"No GR00T config.json found in checkpoint {source}."
        raise FileNotFoundError(msg)


def _complete_ov_config(checkpoint: Path, config_dir: Path) -> None:
    """Add processor files that the Suite exporter only copies from the checkpoint root.

    Fine-tuned checkpoints keep them under ``processor/``; the single-IR loader
    needs them next to ``config.json``.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in _PROCESSOR_FILES:
        target = config_dir / name
        if target.is_file():
            continue
        for candidate in (checkpoint / name, checkpoint / "processor" / name):
            if candidate.is_file():
                shutil.copy2(candidate, target)
                break


def _slices(value: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    for key, bounds in value.items():
        start, end = (bounds["start"], bounds["end"]) if isinstance(bounds, dict) else bounds
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            msg = f"Invalid slice for {key!r}: {bounds!r}."
            raise ValueError(msg)
        slices[str(key)] = (start, end)
    return slices
