# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Package native LeRobot checkpoints and exported LeRobot bundles in the Runtime format."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .policy_support import require_exportable_policy, require_supported_policy, validate_processor_configs

if TYPE_CHECKING:
    from physicalai.engine import Run

_POLICY_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_NATIVE_RESET_CALLBACK = "physicalai_lerobot_policy_plugin.runtime.LeRobotResetCallback"
_BUNDLE_RESET_CALLBACK = "physicalai_lerobot_policy_plugin.bundle_runtime.LeRobotBundleResetCallback"
_ACTION_CHUNK_TRIMMER = "physicalai.inference.postprocessors.ActionChunkTrimmer"
_PACKAGE_METADATA_FILE = "package.json"


def package_lerobot_checkpoint(source: Run | str | Path, output_dir: Path) -> Path:
    """Copy a native LeRobot model directory into a Runtime package.

    The native checkpoint remains under ``<policy>.lerobot/pretrained_model``.
    Runtime locates the ``.lerobot`` directory through its adapter suffix
    lookup; the manifest also declares that artifact for backend detection.

    Args:
        source: Native ``pretrained_model`` directory, a containing checkpoint
            directory, a run output directory, or a LeRobot Engine ``Run``.
        output_dir: New directory where the package will be created.

    Returns:
        The completed Runtime package directory.

    Raises:
        FileExistsError: If *output_dir* already exists.
        ValueError: If the policy type is not supported, or the model name, source tree, or
            saved processors are unsafe or invalid.
    """
    native_dir = _native_model_directory(source)
    policy_name = require_supported_policy(_policy_name(native_dir))
    # Runtime loads these processors; fail here rather than when the package is loaded.
    validate_processor_configs(native_dir)
    output_candidate = Path(output_dir).expanduser()
    if output_candidate.exists() or output_candidate.is_symlink():
        msg = f"Export directory already exists: {output_candidate}"
        raise FileExistsError(msg)
    package_root = output_candidate.resolve()
    if package_root.is_relative_to(native_dir) or native_dir.is_relative_to(package_root):
        msg = "Export directory and native checkpoint must not contain one another."
        raise ValueError(msg)

    package_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{package_root.name}.", dir=package_root.parent) as temporary:
        staged_package = Path(temporary) / "package"
        staged_package.mkdir()
        artifact_name = f"{policy_name}.lerobot"
        staged_native_dir = staged_package / artifact_name / "pretrained_model"
        _copy_native_tree(native_dir, staged_native_dir)
        _write_native_manifest(staged_package, policy_name, artifact_name)
        staged_package.rename(package_root)

    return package_root


def package_lerobot_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    policy_name: str,
    export_backend: str,
    worker_env_key: str | None = None,
    n_action_steps: int | None = None,
    trust_remote_code: bool = False,
) -> Path:
    """Wrap an unmodified ``lerobot-export`` bundle folder in a Physical AI Runtime package.

    The original LeRobot bundle folder is placed verbatim under
    ``<policy>.lerobot-bundle/`` so the same artifact runs under both
    ``InferenceModel(package)`` and upstream ``lerobot-rollout --bundle.path=...``.

    Args:
        bundle_dir: Verified bundle folder produced by ``lerobot-export``.
        output_dir: New directory where the Runtime package will be created.
        policy_name: Validated LeRobot policy type.
        export_backend: Export backend name (``"openvino"`` or ``"executorch_openvino"``).
        worker_env_key: Content key of the isolated environment for Tier 2 provenance checks.
        n_action_steps: Optional ``ActionChunkTrimmer`` postprocessor limit.
        trust_remote_code: Whether *policy_name* was explicitly opted in outside the allowlist.

    Returns:
        The completed Runtime package directory.

    Raises:
        FileExistsError: If *output_dir* already exists.
        FileNotFoundError: If *bundle_dir* does not contain ``export.json``.
        ValueError: If *bundle_dir* contains symlinks or overlaps with *output_dir*.
    """
    validated_policy = require_exportable_policy(policy_name, trust_remote_code=trust_remote_code)
    resolved_bundle = Path(bundle_dir).expanduser().resolve(strict=True)
    if not resolved_bundle.is_dir() or not (resolved_bundle / "export.json").is_file():
        msg = f"LeRobot bundle is missing export.json: {resolved_bundle}"
        raise FileNotFoundError(msg)
    if n_action_steps is not None and (isinstance(n_action_steps, bool) or n_action_steps < 1):
        msg = f"n_action_steps must be a positive integer, got {n_action_steps!r}."
        raise ValueError(msg)

    output_candidate = Path(output_dir).expanduser()
    if output_candidate.exists() or output_candidate.is_symlink():
        msg = f"Export directory already exists: {output_candidate}"
        raise FileExistsError(msg)
    package_root = output_candidate.resolve()
    if package_root.is_relative_to(resolved_bundle) or resolved_bundle.is_relative_to(package_root):
        msg = "Export directory and source bundle must not contain one another."
        raise ValueError(msg)

    export_manifest = _read_json_object(resolved_bundle / "export.json")
    bundle_runtime = export_manifest.get("runtime")
    package_metadata: dict[str, object] = {
        "source_framework": "lerobot",
        "export_backend": export_backend,
        "bundle_runtime": bundle_runtime,
        "policy": validated_policy,
        "tolerance": export_manifest.get("tolerance"),
    }
    if worker_env_key is not None:
        package_metadata["worker_env_key"] = worker_env_key

    package_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{package_root.name}.", dir=package_root.parent) as temporary:
        staged_package = Path(temporary) / "package"
        staged_package.mkdir()
        artifact_name = f"{validated_policy}.lerobot-bundle"
        _copy_native_tree(resolved_bundle, staged_package / artifact_name)
        _write_bundle_manifest(
            staged_package,
            validated_policy,
            artifact_name,
            n_action_steps=n_action_steps,
            extra=package_metadata,
        )
        (staged_package / _PACKAGE_METADATA_FILE).write_text(
            json.dumps(package_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_package.rename(package_root)

    return package_root


def _native_model_directory(source: Run | str | Path) -> Path:
    """Resolve a Run, checkpoint, or run directory to its native model folder.

    Args:
        source: LeRobot Run, checkpoint folder, or model directory.

    Returns:
        Resolved ``pretrained_model`` directory.

    Raises:
        FileNotFoundError: If no native model config is present.
        TypeError: If a Run does not contain a path-like checkpoint.
        ValueError: If a candidate model folder escapes the source directory.
    """
    if isinstance(source, (str, Path)):
        source_path = Path(source).expanduser()
    else:
        if getattr(source, "framework", None) != "lerobot":
            msg = "The source Run must belong to the 'lerobot' framework."
            raise ValueError(msg)
        checkpoint = getattr(source, "checkpoint", None)
        if not isinstance(checkpoint, (str, Path)):
            msg = "The LeRobot Run has no native checkpoint to export."
            raise TypeError(msg)
        source_path = Path(checkpoint).expanduser()

    if source_path.is_file():
        if source_path.name not in {"config.json", "train_config.json"}:
            msg = f"Expected a LeRobot model directory, got file {source_path.name!r}."
            raise ValueError(msg)
        source_path = source_path.parent

    root = source_path.resolve(strict=True)
    if not root.is_dir():
        msg = f"LeRobot checkpoint is not a directory: {root}"
        raise ValueError(msg)

    candidates = (
        root,
        root / "pretrained_model",
        root / "checkpoints" / "last" / "pretrained_model",
    )
    native_dir = next((candidate for candidate in candidates if (candidate / "config.json").is_file()), None)
    if native_dir is None:
        msg = f"No native LeRobot config.json found under {root}."
        raise FileNotFoundError(msg)

    resolved_native_dir = native_dir.resolve(strict=True)
    if not resolved_native_dir.is_relative_to(root):
        msg = "The native LeRobot model directory escapes the supplied checkpoint directory."
        raise ValueError(msg)
    return resolved_native_dir


def _policy_name(native_dir: Path) -> str:
    """Read and validate LeRobot's native policy type from its JSON config.

    Args:
        native_dir: Resolved LeRobot model directory.

    Returns:
        Safe policy type used for the Runtime artifact directory.

    Raises:
        ValueError: If the saved config contains an invalid policy type.
    """
    train_config_path = native_dir / "train_config.json"
    if train_config_path.is_file():
        train_config = _read_json_object(train_config_path)
        if "policy" in train_config:
            policy = train_config["policy"]
            if policy is not None and not isinstance(policy, dict):
                msg = "LeRobot train_config.json field 'policy' must be an object or null."
                raise ValueError(msg)
            if isinstance(policy, dict) and "type" in policy:
                return _validate_policy_name(policy["type"])

    model_config = _read_json_object(native_dir / "config.json")
    if "type" not in model_config:
        msg = "LeRobot config.json is missing its 'type' field."
        raise ValueError(msg)
    return _validate_policy_name(model_config["type"])


def _read_json_object(path: Path) -> dict[str, object]:
    """Read a JSON object from a native LeRobot config file.

    Args:
        path: LeRobot JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the top-level JSON value is not an object.
    """
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        msg = f"Expected a JSON object in {path.name}."
        raise TypeError(msg)
    return value


def _validate_policy_name(value: object) -> str:
    """Validate a LeRobot policy type before using it as a directory name.

    Args:
        value: Untrusted config value for the policy type.

    Returns:
        Safe policy name.

    Raises:
        ValueError: If the value does not match the allowed name pattern.
    """
    if not isinstance(value, str) or _POLICY_NAME_RE.fullmatch(value) is None:
        msg = f"LeRobot policy type must match [a-zA-Z0-9_-]{{1,64}}, got {value!r}."
        raise ValueError(msg)
    return value


def _copy_native_tree(source: Path, destination: Path) -> None:
    """Copy a native model or bundle tree after rejecting symbolic links.

    Args:
        source: Validated directory.
        destination: New package directory for the files.

    Raises:
        ValueError: If the directory contains symbolic links.
    """
    for path in source.rglob("*"):
        if path.is_symlink():
            msg = "LeRobot checkpoint contains symbolic links; materialize the checkpoint before export."
            raise ValueError(msg)
    shutil.copytree(source, destination)


def _write_native_manifest(package_root: Path, policy_name: str, artifact_name: str) -> None:
    """Write Runtime's manifest schema for a native ``.lerobot`` checkpoint package."""
    from physicalai.inference.manifest import (  # ruff: ignore[import-outside-top-level]
        ComponentSpec,
        Manifest,
        MetadataSpec,
        ModelSpec,
        PolicySource,
        PolicySpec,
    )

    manifest = Manifest(
        policy=PolicySpec(name=policy_name, source=PolicySource()),
        model=ModelSpec(
            artifacts={"lerobot": artifact_name},
            callbacks=[ComponentSpec(class_path=_NATIVE_RESET_CALLBACK, init_args={})],
        ),
        metadata=MetadataSpec(created_by="physicalai-lerobot-policy-plugin"),
    )
    manifest.save(package_root / "manifest.json")


def _write_bundle_manifest(
    package_root: Path,
    policy_name: str,
    artifact_name: str,
    *,
    n_action_steps: int | None,
    extra: dict[str, object],
) -> None:
    """Write Runtime's manifest schema for an exported ``.lerobot-bundle`` package."""
    from physicalai.inference.manifest import (  # ruff: ignore[import-outside-top-level]
        ComponentSpec,
        Manifest,
        MetadataSpec,
        ModelSpec,
        PolicySource,
        PolicySpec,
    )

    postprocessors = (
        [ComponentSpec(class_path=_ACTION_CHUNK_TRIMMER, init_args={"n_action_steps": n_action_steps})]
        if n_action_steps is not None
        else []
    )
    manifest = Manifest(
        policy=PolicySpec(name=policy_name, source=PolicySource()),
        model=ModelSpec(
            artifacts={"lerobot_bundle": artifact_name},
            postprocessors=postprocessors,
            callbacks=[ComponentSpec(class_path=_BUNDLE_RESET_CALLBACK, init_args={})],
        ),
        metadata=MetadataSpec(created_by="physicalai-lerobot-policy-plugin"),
        lerobot_bundle=extra,
    )
    manifest.save(package_root / "manifest.json")
