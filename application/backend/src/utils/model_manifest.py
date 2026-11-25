# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light policy manifest rules shared by local and Hub imports."""

from collections.abc import Mapping
from pathlib import PurePosixPath, PureWindowsPath

EXPORT_BACKENDS = ("torch", "openvino", "onnx", "executorch")
SUPPORTED_POLICIES = frozenset({"act", "smolvla", "pi05", "rldx1", "molmoact2", "xr0"})


def manifest_artifact(manifest: Mapping[str, object], backend: str, label: str) -> str:
    """Extract a relative artifact path from a policy manifest."""
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Manifest '{label}' is missing object field 'model'")
    artifacts = model.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Manifest '{label}' is missing object field 'model.artifacts'")
    artifact = artifacts.get(backend)
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError(f"Manifest '{label}' is missing non-empty 'model.artifacts.{backend}' entry")
    path = PurePosixPath(artifact)
    if path.is_absolute() or ".." in path.parts or "\\" in artifact or PureWindowsPath(artifact).drive:
        raise ValueError(f"Manifest '{label}' contains unsafe {backend} artifact path '{artifact}'")
    return artifact


def manifest_policy(manifest: Mapping[str, object], label: str) -> str:
    """Extract a supported policy name without importing policy implementations."""
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"Manifest '{label}' is missing 'policy' section")
    name = policy.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Manifest '{label}' is missing 'policy.name'")
    if name not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Manifest '{label}' declares unsupported policy '{name}'. "
            f"Supported policies are: {', '.join(sorted(SUPPORTED_POLICIES))}"
        )
    return name
