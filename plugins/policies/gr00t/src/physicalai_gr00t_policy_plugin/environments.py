# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime-isolate sources and worker metadata for GR00T environments.

This module is safe to import on Runtime-only hosts. It declares isolate
sources and profiles without importing ``physicalai.engine`` or
``physicalai-train``; the optional training engine adapts these declarations to
its ``WorkerEnv`` and ``Enablement`` types.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from physicalai.isolate import (
    Environment,
    GitProject,
    PatchFile,
    ProjectSource,
    PublishedPatches,
    SyncProfile,
)
from physicalai.isolate import (
    prepare as isolate_prepare,
)

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

from .package import BACKEND as FRAMEWORK
from .package import FORMAT_NATIVE, FORMAT_OPENVINO

DISTRIBUTION = "physicalai-gr00t-policy-plugin"
PROJECT_ENV = "PHYSICALAI_WORKER_PROJECT"
_PACKAGE = resources.files("physicalai_gr00t_policy_plugin")

TRAIN_WORKER = _PACKAGE / "_worker" / "gr00t_train_worker.py"
EXPORT_WORKER = _PACKAGE / "_worker" / "gr00t_export_worker.py"
INFER_WORKER = _PACKAGE / "_worker" / "gr00t_infer_worker.py"

UPSTREAM = GitProject(
    repository="https://github.com/NVIDIA/Isaac-GR00T.git",
    revision="3df8b3825d67f755e69141446f4315f281b9b7e6",
)

ROBOTICS_AI_SUITE_OPENVINO = PublishedPatches(
    repository="https://github.com/open-edge-platform/robotics-ai-suite.git",
    revision="2960959ae6cc6d45150316ed6612f686c2d915b7",
    directory="src/pipelines/gr00t-n1d7-ov/patches",
    files=(
        PatchFile(
            "0001-Add-OpenVINO-model-convert-inference-code.patch",
            "2e7d6355b948926a965ee30df1bc2d7623a51233547d90e8eea16a967188ed2d",
        ),
        PatchFile(
            "0002-Add-lazy-set-static-shape.patch",
            "da7c5266e69362ec4f81bfcf995babfbbeb7f8ff3005fc002d7974137dc86fa0",
        ),
        PatchFile(
            "0003-Add-static-LLM-seq_len-padding-via-llm-lang-tokens.patch",
            "2d47606257fff4214825a8daf82056170c6111dc80f51f674b6c1310eba880e6",
        ),
        PatchFile(
            "0004-Enable-gr00t-n1.7-sonic-pipeline-on-ov.patch",
            "8ea2712b3c3235582a1c491fe81dfda4bd8c4a67be0c37b405f75049e077e912",
        ),
        PatchFile(
            "0005-Refresh-OpenVINO-export-and-inference-READMEs.patch",
            "e91d474388ae3a8ceb815c1f777952dceecb45e862edd77ab578ed08714d9a3d",
        ),
        PatchFile(
            "0006-Enable-3-camera-in-embodiment-tag-UNITREE_G1_SONIC.patch",
            "703fccdef8504b1fe3885b90051fa022e208e2decf97a3a210135d5e37d3e6a3",
        ),
        PatchFile(
            "0007-Address-Bandit-findings.patch",
            "b867bb2e27bfc569457f192b73c79f613f79e4fdf20a3da6bbece41524488227",
        ),
        PatchFile(
            "0008-single-OV-server-path-honor-llm_static_seq_len-paddi.patch",
            "ceb5527f9f558260dccf8617339d24b0e45d4543b228da183e1ec93de489cb09",
        ),
    ),
    tree="d657bcb89b8937495b8c849d7cfb8c9f3d1e5601",
    license="Apache-2.0",
    mirror=_PACKAGE / "patches" / "robotics-ai-suite-gr00t-n1d7-ov",
)


@dataclass(frozen=True)
class WorkerEnvironmentSpec:
    """Runtime-only isolate declaration plus Engine worker metadata."""

    framework: str
    name: str
    environment: Environment
    worker: Traversable
    distribution: str
    package: str
    accelerators: tuple[str, ...]

    @property
    def source(self) -> ProjectSource:
        """Project source consumed by Runtime's isolate provider."""
        return self.environment.source

    @property
    def profile(self) -> SyncProfile:
        """Sync profile selected for this named worker environment."""
        return self.environment.profile

    def key(self) -> str:
        """Return the stable package identity used to verify GR00T environment pins.

        The identity retains the Engine package-key format so artifacts produced
        before this plugin switched to Runtime isolate remain loadable.
        """
        identity = {
            "format": 2,
            "framework": self.framework,
            "name": self.name,
            "python": self.profile.python,
            "extras": {accelerator: list(self.profile.extras) for accelerator in sorted(self.accelerators)},
            "source": self.source.identity(),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


NATIVE = WorkerEnvironmentSpec(
    framework=FRAMEWORK,
    name="default",
    environment=Environment(source=UPSTREAM, profile=SyncProfile(python="3.10")),
    worker=TRAIN_WORKER,
    distribution=DISTRIBUTION,
    package="gr00t",
    accelerators=("cu128",),
)

OPENVINO = WorkerEnvironmentSpec(
    framework=FRAMEWORK,
    name="robotics-ai-suite-openvino",
    environment=Environment(
        source=GitProject(UPSTREAM.repository, UPSTREAM.revision, patches=ROBOTICS_AI_SUITE_OPENVINO),
        profile=SyncProfile(python="3.12"),
    ),
    worker=EXPORT_WORKER,
    distribution=DISTRIBUTION,
    package="gr00t",
    accelerators=("cpu",),
)

ENVIRONMENTS: dict[str, WorkerEnvironmentSpec] = {NATIVE.name: NATIVE, OPENVINO.name: OPENVINO}
PACKAGE_ENVIRONMENTS: dict[str, WorkerEnvironmentSpec] = {FORMAT_NATIVE: NATIVE, FORMAT_OPENVINO: OPENVINO}


def resolve(
    env: WorkerEnvironmentSpec,
    *,
    interpreter: str | Path | None = None,
    project: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    """Return the interpreter and checkout for *env*, preparing it when needed.

    Args:
        env: Runtime-isolate environment declaration.
        interpreter: User-managed worker interpreter that bypasses provisioning.
        project: Checkout used by *interpreter*; defaults to ``$PHYSICALAI_WORKER_PROJECT``.

    Returns:
        Interpreter, upstream checkout, and whether isolate manages the environment.

    Raises:
        ValueError: If a user-managed interpreter has no checkout.
    """
    if interpreter is None:
        prepared = isolate_prepare(env.environment)
        return prepared.python, prepared.project, True
    checkout = project if project is not None else os.environ.get(PROJECT_ENV)
    if not checkout:
        msg = f"Pass project= or set {PROJECT_ENV} to the Isaac-GR00T checkout used by {interpreter}."
        raise ValueError(msg)
    return Path(interpreter).expanduser(), Path(checkout).expanduser(), False


def locked_version(env: WorkerEnvironmentSpec, project: Path) -> str:
    """Read the exact GR00T version from a prepared isolate project lock.

    Returns:
        The version pinned for the GR00T distribution.

    Raises:
        ValueError: If the lock is invalid or does not pin exactly one GR00T package.
    """
    lock_path = project / "uv.lock"
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        msg = f"Could not read the GR00T environment lock at {lock_path}."
        raise ValueError(msg) from exc
    package_name = _normalize_distribution_name(env.package)
    versions = {
        package["version"]
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and isinstance(package.get("name"), str)
        and _normalize_distribution_name(package["name"]) == package_name
        and isinstance(package.get("version"), str)
    }
    if len(versions) != 1:
        msg = f"Expected one locked version of {env.package!r}, found {sorted(versions)}."
        raise ValueError(msg)
    return next(iter(versions))


def provenance(env: WorkerEnvironmentSpec) -> dict[str, object]:
    """Describe an environment for package and run records.

    Returns:
        Environment name, stable package identity key, and source provenance.
    """
    return {"framework": env.framework, "name": env.name, "key": env.key(), "source": env.source.provenance()}


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
