# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for Runtime-owned worker project sources."""

from __future__ import annotations

from pathlib import Path

from physicalai import isolate
from physicalai.engine import (
    BundledPatches,
    BundledProject,
    GitProject,
    PatchFile,
    PatchSeries,
    ProjectSource,
    PublishedPatches,
    WorkerEnv,
)


def test_engine_source_types_are_runtime_isolate_types() -> None:
    assert BundledPatches is isolate.BundledPatches
    assert BundledProject is isolate.BundledProject
    assert GitProject is isolate.GitProject
    assert PatchFile is isolate.PatchFile
    assert PatchSeries is isolate.PatchSeries
    assert ProjectSource is isolate.ProjectSource
    assert PublishedPatches is isolate.PublishedPatches


def test_worker_environment_wraps_a_project_directory_with_runtime_source(tmp_path: Path) -> None:
    project = tmp_path / "worker"
    project.mkdir()
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n", encoding="utf-8")

    environment = WorkerEnv(
        framework="demo",
        project=project,
        worker=worker,
        distribution="demo-plugin",
        package="demo-worker",
    )

    assert isinstance(environment.source, isolate.BundledProject)
    assert environment.source.identity() == isolate.BundledProject(project).identity()
