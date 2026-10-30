# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only detection shared by CLI inspection and import."""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

BACKENDS = ("torch", "openvino", "onnx", "executorch")
POLICIES = {"act", "smolvla", "pi05", "rldx1", "molmoact2", "xr0"}
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


def hub_api() -> Any:
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


def _metadata(read_json: Callable[[str], dict], files: set[str], kind: str) -> dict:
    if "studio.json" not in files:
        return {}
    data = read_json("studio.json")
    if data.get("schema_version") != 1 or data.get("kind") != kind:
        raise ValueError("Unsupported studio.json schema version or resource kind")
    payload = data.get("files")
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("studio.json must contain a payload file list")
    for name in payload:
        safe_path(name)
    if not isinstance(data.get("name"), str) or not data["name"].strip() or len(data["name"]) > 255:
        raise ValueError("Studio asset name must contain 1 to 255 characters")
    return data


def inspect_model_files(  # noqa: C901, PLR0912, PLR0915
    sizes: dict[str, int | None],
    read_json: Callable[[str], dict],
    subfolder: str | None = None,
) -> dict[str, Any]:
    """Detect and validate Studio model files without loading a policy."""
    try:
        for name in sizes:
            safe_path(name)
        root = _model_root(set(sizes), subfolder)
    except ValueError as exc:
        return {
            "schema_version": 1,
            "kind": "model",
            "subfolder": subfolder,
            "validation_level": "metadata",
            "importable": False,
            "errors": [str(exc)],
            "warnings": [],
            "files": [],
            "size_bytes": 0,
            "size_complete": True,
        }
    prefix = f"{root}/" if root else ""
    relative = {name.removeprefix(prefix): size for name, size in sizes.items() if name.startswith(prefix)}
    files = set(relative)

    def read(name: str) -> dict:
        return read_json(f"{prefix}{safe_path(name)}")

    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "model",
        "subfolder": root,
        "validation_level": "metadata",
        "warnings": warnings,
        "errors": errors,
    }
    try:
        metadata = _metadata(read, files, "model")
        result["studio"] = metadata
        if metadata:
            if metadata.get("status") != "complete":
                errors.append("This Studio export is incomplete. Resume its export before importing it.")
            missing = set(metadata["files"]) - files
            if missing:
                errors.append(f"Studio payload is incomplete; missing {sorted(missing)}")
        if type(metadata.get("version", 1)) is not int or metadata.get("version", 1) < 1:
            raise ValueError("Studio model version must be a positive integer")
        properties = metadata.get("properties", {})
        if not isinstance(properties, dict) or any(
            not isinstance(properties[key], bool)
            for key in ("lora_enabled", "lora_use_dora", "snapflow_enabled")
            if key in properties
        ):
            raise ValueError("Studio model properties must contain boolean feature flags")
        backends = {}
        policies = set()
        exports_only = not any(f"exports/{backend}/manifest.json" in files for backend in BACKENDS)
        for backend in BACKENDS:
            directory = backend if exports_only else f"exports/{backend}"
            manifest_name = f"{directory}/manifest.json"
            if manifest_name not in files:
                continue
            manifest = read(manifest_name)
            policy = manifest.get("policy", {}).get("name")
            if manifest.get("format") != "policy_package" or manifest.get("version") != "1.0":
                errors.append(f"Unsupported policy manifest: {manifest_name}")
            if policy not in POLICIES:
                errors.append(f"Unsupported policy: {policy}")
            policies.add(policy)
            artifact = manifest.get("model", {}).get("artifacts", {}).get(backend)
            artifact = safe_path(artifact) if isinstance(artifact, str) else ""
            if not artifact or f"{directory}/{artifact}" not in files:
                errors.append(f"Missing {backend} artifact referenced by {manifest_name}")
            if backend == "openvino" and artifact.endswith(".xml"):
                for xml in [name for name in files if name.startswith(f"{directory}/") and name.endswith(".xml")]:
                    if f"{xml[:-4]}.bin" not in files:
                        errors.append(f"Missing OpenVINO companion {xml[:-4]}.bin")
            backends[backend] = manifest
        if not backends or len(policies) != 1:
            errors.append("Expected supported export manifests with one consistent policy")
        if not exports_only and "model.ckpt" not in files:
            warnings.append("Deployment-only model: no model.ckpt is available to resume training.")
        if "snapflow-model.ckpt" in files:
            warnings.append("SnapFlow checkpoint is present; original logger hparams may describe the base phase.")
        if metadata.get("policy") is not None and metadata["policy"] not in policies:
            errors.append("Studio metadata policy disagrees with export manifests")
        result.update(
            detected_format="studio_exports" if exports_only else "studio_model",
            model={
                "policy": next(iter(policies), None),
                "backends": backends,
                "checkpoints": sorted(name for name in files if name.endswith(".ckpt")),
            },
            required_import_options=["project_id", "dataset_id"],
        )
        prefixes = tuple(f"{backend}/" for backend in BACKENDS) if exports_only else ("exports/",)
        payload = [
            name
            for name in files
            if name.startswith(prefixes)
            or name in {"model.ckpt", "snapflow-model.ckpt", "version_0/hparams.yaml", "version_0/metrics.csv"}
        ]
        payload += [name for name in files if name in {"README.md", "LICENSE", "LICENSE.md", "studio.json"}]
        undeclared = set(metadata.get("files", [])) - set(payload)
        if undeclared:
            errors.append(f"Studio metadata references files outside the supported layout: {sorted(undeclared)}")
        result["files"] = sorted(payload)
        result["size_bytes"] = sum(relative[name] or 0 for name in payload)
        result["size_complete"] = all(relative[name] is not None for name in payload)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(str(exc))
    result["importable"] = not errors
    return result


def inspect_dataset_files(  # noqa: C901, PLR0912
    sizes: dict[str, int | None],
    read_json: Callable[[str], dict],
    subfolder: str | None = None,
) -> dict[str, Any]:
    """Detect and validate LeRobot v3 metadata without downloading data shards."""
    try:
        for name in sizes:
            safe_path(name)
        root = _dataset_root(set(sizes), subfolder)
    except ValueError as exc:
        return {
            "schema_version": 1,
            "kind": "dataset",
            "subfolder": subfolder,
            "validation_level": "metadata",
            "importable": False,
            "errors": [str(exc)],
            "warnings": [],
            "files": [],
            "size_bytes": 0,
            "size_complete": True,
        }
    prefix = f"{root}/" if root else ""
    relative = {name.removeprefix(prefix): size for name, size in sizes.items() if name.startswith(prefix)}
    files = set(relative)

    def read(name: str) -> dict:
        return read_json(f"{prefix}{safe_path(name)}")

    errors: list[str] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dataset",
        "subfolder": root,
        "validation_level": "metadata",
        "warnings": [],
        "errors": errors,
    }
    try:
        metadata = _metadata(read, files, "dataset")
        result["studio"] = metadata
        if metadata:
            if metadata.get("status") != "complete":
                errors.append("This Studio export is incomplete. Resume its export before importing it.")
            missing = set(metadata["files"]) - files
            if missing:
                errors.append(f"Studio payload is incomplete; missing {sorted(missing)}")
        if not isinstance(metadata.get("default_task", ""), str):
            raise ValueError("Studio dataset default_task must be a string")
        info = read("meta/info.json")
        if info.get("codebase_version") != "v3.0":
            errors.append("Only LeRobot v3.0 datasets are supported")
        for marker in ("meta/tasks.parquet", "meta/stats.json"):
            if marker not in files:
                errors.append(f"Missing {marker}")
        for pattern in ("data/chunk-*/file-*.parquet", "meta/episodes/chunk-*/file-*.parquet"):
            if not any(PurePosixPath(name).match(pattern) for name in files):
                errors.append(f"Missing {pattern}")
        features = info.get("features", {})
        if not isinstance(features, dict):
            raise ValueError("Dataset features must be an object")
        cameras = [name for name, feature in features.items() if feature.get("dtype") == "video"]
        for camera in cameras:
            if not any(name.startswith(f"videos/{camera}/") and name.endswith(".mp4") for name in files):
                errors.append(f"No video files for {camera}")
        result.update(
            detected_format="lerobot_v3",
            dataset={
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
                "features": features,
                "cameras": cameras,
            },
            required_import_options=["project_id"],
        )
        payload = [name for name in files if name.startswith(("meta/", "data/", "videos/"))]
        payload += [name for name in files if name in {"README.md", "LICENSE", "LICENSE.md", "studio.json"}]
        undeclared = set(metadata.get("files", [])) - set(payload)
        if undeclared:
            errors.append(f"Studio metadata references files outside the supported layout: {sorted(undeclared)}")
        result["files"] = sorted(payload)
        result["size_bytes"] = sum(relative[name] or 0 for name in payload)
        result["size_complete"] = all(relative[name] is not None for name in payload)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        errors.append(str(exc))
    result["importable"] = not errors
    return result


def _remote_files(repo_type: str, repo_id: str, revision: str) -> tuple[str, dict, Callable[[str], dict]]:
    """Read a pinned inventory and provide a size-bounded JSON download function."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    api = hub_api()
    info = api.repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    sizes = {item.rfilename: item.size for item in info.siblings}

    def read(name: str) -> dict:
        if name not in sizes or sizes[name] is None or sizes[name] > MAX_METADATA_BYTES:
            raise ValueError(f"Missing or oversized metadata: {name}")
        path = Path(hf_hub_download(repo_id, name, repo_type=repo_type, revision=info.sha, token=token()))
        return read_json_file(path)

    return info.sha, sizes, read


def inspect_remote_model(repo_id: str, revision: str = "main", subfolder: str | None = None) -> dict[str, Any]:
    """Inspect a pinned Studio model repository."""
    sha, sizes, read = _remote_files("model", repo_id, revision)
    return {**inspect_model_files(sizes, read, subfolder), "repo_id": repo_id, "revision": sha}


def inspect_remote_dataset(repo_id: str, revision: str = "main", subfolder: str | None = None) -> dict[str, Any]:
    """Inspect a pinned LeRobot dataset repository."""
    sha, sizes, read = _remote_files("dataset", repo_id, revision)
    return {**inspect_dataset_files(sizes, read, subfolder), "repo_id": repo_id, "revision": sha}


def read_json_file(path: Path) -> dict:
    """Read bounded JSON without constructing objects specified by the resource."""
    with path.open("rb") as stream:
        content = stream.read(MAX_METADATA_BYTES + 1)
    if len(content) > MAX_METADATA_BYTES:
        raise ValueError(f"Metadata exceeds {MAX_METADATA_BYTES} bytes: {path.name}")
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return result


def _local_sizes(root: Path) -> dict[str, int | None]:
    sizes: dict[str, int | None] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {path}")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            sizes[path.relative_to(root).as_posix()] = path.stat().st_size
    return sizes


def inspect_local_model(root: Path) -> dict[str, Any]:
    """Inspect local model files, rejecting symbolic links."""
    return inspect_model_files(_local_sizes(root), lambda name: read_json_file(root / name), subfolder="")


def inspect_local_dataset(root: Path) -> dict[str, Any]:
    """Inspect local dataset files, rejecting symbolic links."""
    return inspect_dataset_files(_local_sizes(root), lambda name: read_json_file(root / name), subfolder="")
