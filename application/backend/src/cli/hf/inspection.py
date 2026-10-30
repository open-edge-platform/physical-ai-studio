# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only detection shared by CLI inspection and import."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from cli.hf.schemas import (
    AssetMetadata,
    DatasetDetails,
    DatasetInfo,
    Inspection,
    JsonObject,
    ModelDetails,
    ResourceKind,
)
from utils.model_manifest import EXPORT_BACKENDS as BACKENDS
from utils.model_manifest import manifest_artifact, manifest_policy

if TYPE_CHECKING:
    from collections.abc import Callable

    from huggingface_hub import DatasetInfo as HubDatasetInfo
    from huggingface_hub import HfApi, ModelInfo

CARDS_AND_LICENSES = {"README.md", "LICENSE", "LICENSE.md"}
MAX_METADATA_BYTES = 4 * 1024 * 1024


def token() -> str | bool:
    """Resolve Studio settings, then HF_TOKEN; otherwise use anonymous access."""
    from settings import get_settings

    # Match the training resolver's precedence without importing services/__init__,
    # which initializes unrelated training, cloud, and hardware dependencies.
    secret = get_settings().huggingface.hf_token
    return (secret.get_secret_value() if secret else None) or os.environ.get("HF_TOKEN") or False


def redact(message: str) -> str:
    """Remove configured credentials and recognizable Hub tokens from diagnostics."""
    credential = token()
    if isinstance(credential, str):
        message = message.replace(credential, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)


def hub_api() -> HfApi:
    """Construct an authenticated client without changing global login state."""
    from huggingface_hub import HfApi

    return HfApi(token=token())


def safe_path(value: str, *, allow_root: bool = False) -> str:
    """Validate a portable relative Hub path before using it on disk."""
    if allow_root and value == "":
        return ""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or "\\" in value
        or ":" in value
        or any(char in value for char in "*?[]")
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"Unsafe resource path: {value!r}")
    return value


def _model_root(files: set[str], subfolder: str | None) -> str:
    if subfolder is not None:
        return safe_path(subfolder, allow_root=True)
    candidates: set[str] = set()
    for name in files:
        if name == "studio.json" or name.endswith("/studio.json"):
            candidates.add(name.removesuffix("studio.json").rstrip("/"))
        for backend in BACKENDS:
            suffix = f"{backend}/manifest.json"
            if name == suffix or name.endswith(f"/{suffix}"):
                parent = name.removesuffix(suffix).rstrip("/")
                if parent == "exports" or parent.endswith("/exports"):
                    parent = parent.removesuffix("exports").rstrip("/")
                candidates.add(parent)
    return _one_root(candidates, "model")


def _dataset_root(files: set[str], subfolder: str | None) -> str:
    if subfolder is not None:
        return safe_path(subfolder, allow_root=True)
    candidates = set()
    for name in files:
        for suffix in ("studio.json", "meta/info.json"):
            if name == suffix or name.endswith(f"/{suffix}"):
                candidates.add(name.removesuffix(suffix).rstrip("/"))
    return _one_root(candidates, "dataset")


def _one_root(candidates: set[str], resource: str) -> str:
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {resource} root, found {sorted(candidates)}. "
            "Use --subfolder PATH to select an asset; only Studio models and LeRobot v3 datasets are supported."
        )
    return candidates.pop()


@dataclass(frozen=True)
class _Inventory:
    sizes: dict[str, int | None]
    read: Callable[[str], JsonObject]


def _select_inventory(sizes: dict[str, int | None], read_json: Callable[[str], JsonObject], root: str) -> _Inventory:
    for name in sizes:
        safe_path(name)
    prefix = f"{root}/" if root else ""
    relative = {name.removeprefix(prefix): size for name, size in sizes.items() if name.startswith(prefix)}

    def read(name: str) -> JsonObject:
        return read_json(f"{prefix}{safe_path(name)}")

    return _Inventory(relative, read)


def _metadata(
    read_json: Callable[[str], JsonObject], files: set[str], kind: ResourceKind, errors: list[str]
) -> AssetMetadata:
    if "studio.json" not in files:
        return AssetMetadata()
    data = read_json("studio.json")
    if data.get("schema_version") != 1 or data.get("kind") != kind:
        raise ValueError("Unsupported studio.json schema version or resource kind")
    if "files" not in data:
        raise ValueError("studio.json must contain a payload file list")
    metadata = AssetMetadata.model_validate(data)
    for name in metadata.files:
        safe_path(name)
    if metadata.name is None:
        raise ValueError("Studio asset name must contain 1 to 255 characters")
    if metadata.status != "complete":
        errors.append("This Studio export is incomplete. Resume its export before importing it.")
    missing = set(metadata.files) - files
    if missing:
        errors.append(f"Studio payload is incomplete; missing {sorted(missing)}")
    return metadata


def _set_payload(result: Inspection, inventory: _Inventory, metadata: AssetMetadata, payload: set[str]) -> None:
    payload |= inventory.sizes.keys() & (CARDS_AND_LICENSES | {"studio.json"})
    undeclared = set(metadata.files) - payload
    if undeclared:
        result.errors.append(f"Studio metadata references files outside the supported layout: {sorted(undeclared)}")
    result.files = sorted(payload)
    result.size_bytes = sum(inventory.sizes[name] or 0 for name in payload)
    result.size_complete = all(inventory.sizes[name] is not None for name in payload)


def _model_manifests(
    inventory: _Inventory, exports_only: bool, errors: list[str]
) -> tuple[dict[str, JsonObject], set[str]]:
    files = set(inventory.sizes)
    backends: dict[str, JsonObject] = {}
    policies: set[str] = set()
    for backend in BACKENDS:
        directory = backend if exports_only else f"exports/{backend}"
        manifest_name = f"{directory}/manifest.json"
        if manifest_name not in files:
            continue
        try:
            manifest = inventory.read(manifest_name)
            if manifest.get("format") != "policy_package" or manifest.get("version") != "1.0":
                errors.append(f"Unsupported policy manifest: {manifest_name}")
            policies.add(manifest_policy(manifest, manifest_name))
            artifact = safe_path(manifest_artifact(manifest, backend, manifest_name))
            if f"{directory}/{artifact}" not in files:
                errors.append(f"Missing {backend} artifact referenced by {manifest_name}")
            if backend == "openvino" and artifact.endswith(".xml"):
                errors.extend(
                    f"Missing OpenVINO companion {name[:-4]}.bin"
                    for name in sorted(files)
                    if name.startswith(f"{directory}/") and name.endswith(".xml") and f"{name[:-4]}.bin" not in files
                )
            backends[backend] = manifest
        except ValueError as exc:
            errors.append(str(exc))
    if not backends or len(policies) != 1:
        errors.append("Expected supported export manifests with one consistent policy")
    return backends, policies


def inspect_model_files(
    sizes: dict[str, int | None], read_json: Callable[[str], JsonObject], subfolder: str | None = None
) -> Inspection:
    """Detect and validate Studio model files without loading a policy."""
    result = Inspection(kind="model", subfolder=subfolder)
    try:
        root = _model_root(set(sizes), subfolder)
        inventory = _select_inventory(sizes, read_json, root)
        result.subfolder = root
        files = set(inventory.sizes)
        metadata = _metadata(inventory.read, files, "model", result.errors)
        result.studio = metadata
        exports_only = not any(f"exports/{backend}/manifest.json" in files for backend in BACKENDS)
        backends, policies = _model_manifests(inventory, exports_only, result.errors)
        if not exports_only and "model.ckpt" not in files:
            result.warnings.append("Deployment-only model: no model.ckpt is available to resume training.")
        if "snapflow-model.ckpt" in files:
            result.warnings.append(
                "SnapFlow checkpoint is present; original logger hparams may describe the base phase."
            )
        if metadata.policy is not None and metadata.policy not in policies:
            result.errors.append("Studio metadata policy disagrees with export manifests")
        result.detected_format = "studio_exports" if exports_only else "studio_model"
        result.model = ModelDetails(
            policy=next(iter(sorted(policies)), None),
            backends=backends,
            checkpoints=sorted(name for name in files if name.endswith(".ckpt")),
        )
        result.required_import_options = ["project_id", "dataset_id"]
        prefixes = tuple(f"{backend}/" for backend in BACKENDS) if exports_only else ("exports/",)
        payload = {
            name
            for name in files
            if name.startswith(prefixes)
            or name in {"model.ckpt", "snapflow-model.ckpt", "version_0/hparams.yaml", "version_0/metrics.csv"}
        }
        _set_payload(result, inventory, metadata, payload)
    except ValueError as exc:
        result.errors.append(str(exc))
    return result


def _dataset_info(inventory: _Inventory, errors: list[str]) -> DatasetDetails:
    files = set(inventory.sizes)
    info = DatasetInfo.model_validate(inventory.read("meta/info.json"))
    if info.codebase_version != "v3.0":
        errors.append("Only LeRobot v3.0 datasets are supported")
    for marker in ("meta/tasks.parquet", "meta/stats.json"):
        if marker not in files:
            errors.append(f"Missing {marker}")
    for pattern in ("data/chunk-*/file-*.parquet", "meta/episodes/chunk-*/file-*.parquet"):
        if not any(PurePosixPath(name).match(pattern) for name in files):
            errors.append(f"Missing {pattern}")
    cameras = info.cameras
    for camera in cameras:
        if not any(name.startswith(f"videos/{camera}/") and name.endswith(".mp4") for name in files):
            errors.append(f"No video files for {camera}")
    return DatasetDetails(
        episodes=info.total_episodes,
        frames=info.total_frames,
        fps=info.fps,
        robot_type=info.robot_type,
        features=info.features,
        cameras=cameras,
    )


def inspect_dataset_files(
    sizes: dict[str, int | None], read_json: Callable[[str], JsonObject], subfolder: str | None = None
) -> Inspection:
    """Detect and validate LeRobot v3 metadata without downloading data shards."""
    result = Inspection(kind="dataset", subfolder=subfolder)
    try:
        root = _dataset_root(set(sizes), subfolder)
        inventory = _select_inventory(sizes, read_json, root)
        result.subfolder = root
        files = set(inventory.sizes)
        metadata = _metadata(inventory.read, files, "dataset", result.errors)
        result.studio = metadata
        result.dataset = _dataset_info(inventory, result.errors)
        result.detected_format = "lerobot_v3"
        result.required_import_options = ["project_id"]
        payload = {name for name in files if name.startswith(("meta/", "data/", "videos/"))}
        _set_payload(result, inventory, metadata, payload)
    except ValueError as exc:
        result.errors.append(str(exc))
    return result


def repo_inventory(
    api: HfApi, kind: ResourceKind, repo_id: str, revision: str | None = None
) -> tuple[str, dict[str, int | None]]:
    """Require a pinned model/dataset inventory from the SDK's broader repo API."""
    info = cast(
        "ModelInfo | HubDatasetInfo", api.repo_info(repo_id, repo_type=kind, revision=revision, files_metadata=True)
    )
    if not info.sha or info.siblings is None:
        raise ValueError("Hub did not return a pinned file inventory")
    return info.sha, {item.rfilename: item.size for item in info.siblings}


def _remote_files(
    repo_type: ResourceKind, repo_id: str, revision: str
) -> tuple[str, dict[str, int | None], Callable[[str], JsonObject]]:
    """Read a pinned inventory and provide a size-bounded JSON download function."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    sha, sizes = repo_inventory(hub_api(), repo_type, repo_id, revision)

    def read(name: str) -> JsonObject:
        size = sizes.get(name)
        if size is None or size > MAX_METADATA_BYTES:
            raise ValueError(f"Missing or oversized metadata: {name}")
        path = Path(hf_hub_download(repo_id, name, repo_type=repo_type, revision=sha, token=token()))
        return read_json_file(path)

    return sha, sizes, read


def inspect_remote_model(repo_id: str, revision: str = "main", subfolder: str | None = None) -> Inspection:
    """Inspect a pinned Studio model repository."""
    sha, sizes, read = _remote_files("model", repo_id, revision)
    result = inspect_model_files(sizes, read, subfolder)
    result.repo_id, result.revision = repo_id, sha
    return result


def inspect_remote_dataset(repo_id: str, revision: str = "main", subfolder: str | None = None) -> Inspection:
    """Inspect a pinned LeRobot dataset repository."""
    sha, sizes, read = _remote_files("dataset", repo_id, revision)
    result = inspect_dataset_files(sizes, read, subfolder)
    result.repo_id, result.revision = repo_id, sha
    return result


def read_json_file(path: Path) -> JsonObject:
    """Read bounded JSON without constructing objects specified by the resource."""
    with path.open("rb") as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError(f"Metadata exceeds {MAX_METADATA_BYTES} bytes: {path.name}")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return cast("JsonObject", result)


def _local_sizes(root: Path) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {path}")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            sizes[path.relative_to(root).as_posix()] = path.stat().st_size
    return sizes


def inspect_local_model(root: Path) -> Inspection:
    """Inspect local model files, rejecting symbolic links."""
    return inspect_model_files(_local_sizes(root), lambda name: read_json_file(root / name), subfolder="")


def inspect_local_dataset(root: Path) -> Inspection:
    """Inspect local dataset files, rejecting symbolic links."""
    return inspect_dataset_files(_local_sizes(root), lambda name: read_json_file(root / name), subfolder="")
