# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Isolated worker environment declarations for the LeRobot plugin.

This module uses only ``physicalai.isolate`` for Runtime-side environment helpers so
that Runtime-only installs (without ``physicalai-train``) can still prepare the Tier 2
export/ExecuTorch environment. Engine-side ``WorkerEnv`` declarations are constructed
lazily via :func:`training_worker_env` and :func:`export_worker_env`.
"""

from __future__ import annotations

from importlib import resources
from types import MappingProxyType
from typing import TYPE_CHECKING

from physicalai import isolate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable

    from physicalai.engine import WorkerEnv

PLUGIN_DISTRIBUTION = "physicalai-lerobot-policy-plugin"
PLUGIN_ROOT = resources.files(__package__)
LEROBOT_EXPORT_UPSTREAM_REV = "4e618de14b4a22e87a18d93f8ffe79bfa6efd6bc"
LEROBOT_EXPORT_UPSTREAM_REF = f"https://github.com/shoumikhin/lerobot@{LEROBOT_EXPORT_UPSTREAM_REV}"

_TRAINING_ACCELERATOR_EXTRAS = {"cpu": "cpu", "cuda": "cu132", "xpu": "xpu"}

# Policy types that require a policy-family uv extra in envs/default/ and envs/export/.
POLICY_FAMILY_EXTRAS: Mapping[str, str] = MappingProxyType(
    {
        "smolvla": "smolvla",
        "pi0": "pi",
        "pi05": "pi",
        "pi0_fast": "pi",
        "fineart_vla": "pi",
        "molmoact2": "molmoact2",
        "groot": "vla",
        "xvla": "vla",
        "g05": "vla",
        "cosmos3_edge": "vla",
        "lingbot_va": "vla",
        "multi_task_dit": "vla",
        "evo1": "vla",
        "eo1": "vla",
        "vla_jepa": "vla",
        "fastwam": "vla",
        "wall_oss_05": "vla",
        "dm05": "vla",
        "lawam": "vla",
        "flux3": "vla",
        "wall_x": "vla",
        "lingbot_vla_v2": "vla",
    },
)


def default_project_resource() -> Traversable:
    """Return the bundled stock LeRobot training project resource."""
    return PLUGIN_ROOT / "envs" / "default"


def export_project_resource() -> Traversable:
    """Return the bundled LeRobot export project resource."""
    return PLUGIN_ROOT / "envs" / "export"


def openvino_enablement_resource() -> Traversable:
    """Return the bundled temporary OpenVINO backend/runtime enablement directory."""
    return PLUGIN_ROOT / "_enablements" / "openvino"


def train_worker_resource() -> Traversable:
    """Return the training worker entrypoint resource."""
    return PLUGIN_ROOT / "_worker" / "train.py"


def export_worker_resource() -> Traversable:
    """Return the export worker entrypoint resource."""
    return PLUGIN_ROOT / "_worker" / "export.py"


def infer_worker_resource() -> Traversable:
    """Return the Tier 2 bundle inference worker entrypoint resource."""
    return PLUGIN_ROOT / "_worker" / "infer.py"


def export_isolate_environment(family: str | None = None) -> isolate.Environment:
    """Build the Runtime-owned ``isolate.Environment`` for the export/ExecuTorch env.

    Works in Runtime-only installations without ``physicalai-train``.
    """
    extras = ("cpu", family) if family else ("cpu",)
    return isolate.Environment(
        source=isolate.BundledProject(export_project_resource()),
        profile=isolate.SyncProfile(extras=extras),
    )


def training_worker_env(name: str = "default") -> WorkerEnv:
    """Declare a stock LeRobot training ``WorkerEnv`` from ``envs/default/``."""
    from physicalai.engine import WorkerEnv

    family = () if name == "default" else (name,)
    return WorkerEnv(
        framework="lerobot",
        project=isolate.BundledProject(default_project_resource()),
        worker=train_worker_resource(),
        distribution=PLUGIN_DISTRIBUTION,
        package="lerobot",
        name=name,
        extras={accelerator: (extra, *family) for accelerator, extra in _TRAINING_ACCELERATOR_EXTRAS.items()},
    )


def export_worker_env(family: str | None = None) -> WorkerEnv:
    """Declare a LeRobot export ``WorkerEnv`` from ``envs/export/``."""
    from physicalai.engine import WorkerEnv

    name = "export" if family is None else f"export-{family}"
    extras = ("cpu",) if family is None else ("cpu", family)
    return WorkerEnv(
        framework="lerobot",
        project=isolate.BundledProject(export_project_resource()),
        worker=export_worker_resource(),
        distribution=PLUGIN_DISTRIBUTION,
        package="lerobot",
        accelerators=("cpu",),
        name=name,
        extras={"cpu": extras},
    )


def export_family_for_policy(policy_type: str | None) -> str | None:
    """Return the uv extra family required to export *policy_type*, or ``None`` for base."""
    if policy_type is None:
        return None
    return POLICY_FAMILY_EXTRAS.get(policy_type)
