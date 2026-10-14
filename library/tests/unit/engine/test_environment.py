# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Managed worker environment behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from physicalai.engine import WorkerEnv
from physicalai.engine import _environment


def _worker_env(project: Path, *, framework: str = "demo") -> WorkerEnv:
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text("[project]\nname='demo-worker'\n", encoding="utf-8")
    (project / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "lerobot"\nversion = "0.6.1"\n',
        encoding="utf-8",
    )
    worker = project.parent / "worker.py"
    worker.write_text("# worker\n", encoding="utf-8")
    return WorkerEnv(
        framework=framework,
        project=project,
        worker=worker,
        distribution="demo-plugin",
        package="lerobot",
    )


def test_prepare_copies_the_locked_project_and_syncs_the_selected_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = _worker_env(tmp_path / "source")
    monkeypatch.setenv("PHYSICALAI_ENGINE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(_environment, "uv_bin", lambda: "/tools/uv")
    calls: list[tuple[list[str], dict[str, str], Path]] = []

    def fake_run(
        command: list[str], *, check: bool, cwd: Path, env: dict[str, str], shell: bool
    ) -> None:
        assert check
        assert not shell
        calls.append((command, env, cwd))

    monkeypatch.setattr(_environment.subprocess, "run", fake_run)

    python = _environment.prepare(env, accelerator="cuda", offline=True)

    assert python.name == "python"
    assert (python.parent.parent.parent / "project" / "uv.lock").read_text(encoding="utf-8") == (
        env.project / "uv.lock"
    ).read_text(encoding="utf-8")
    command, command_env, cwd = calls[0]
    assert command == [
        "/tools/uv",
        "sync",
        "--frozen",
        "--project",
        str(cwd),
        "--extra",
        "cu128",
        "--offline",
    ]
    assert cwd.name == "project"
    assert command_env["UV_PROJECT_ENVIRONMENT"] == str(python.parent.parent)
    assert (
        json.loads((python.parent.parent.parent / ".physicalai-engine.json").read_text(encoding="utf-8"))["framework"]
        == "demo"
    )


def test_accelerator_auto_prefers_xpu_then_cuda_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _environment,
        "_torch_accelerators",
        lambda: {"xpu": False, "cuda": True},
    )
    assert _environment.resolve_accelerator("auto") == "cu128"
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": True, "cuda": True})
    assert _environment.resolve_accelerator("auto") == "xpu"
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    assert _environment.resolve_accelerator("auto") == "cpu"

    with pytest.raises(ValueError, match="Unsupported accelerator"):
        _environment.resolve_accelerator("mps")


def test_worker_lock_and_plugin_metadata_are_the_version_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = _worker_env(tmp_path / "project")
    monkeypatch.setattr(
        _environment.metadata,
        "requires",
        lambda distribution: ["lerobot[dataset]>=0.6.1,<0.7"],
    )

    assert env.package == "lerobot"
    assert str(env.pinned()) == "0.6.1"
    assert env.supported().contains("0.6.1")
    assert not env.supported().contains("0.7.0")
    assert env.lock_hash() == _environment.lock_hash(env)


def test_prune_removes_only_stale_managed_worker_environments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    current = cache / "current"
    stale = cache / "stale"
    unrelated = cache / "unrelated"
    for directory in (current, stale, unrelated):
        directory.mkdir()
    marker = {
        "framework": "demo",
        "accelerator": "cpu",
        "lock_hash": "current-lock",
        "distribution": "demo-plugin",
        "format_version": 1,
    }
    (current / ".physicalai-engine.json").write_text(json.dumps(marker), encoding="utf-8")
    (stale / ".physicalai-engine.json").write_text(
        json.dumps({**marker, "lock_hash": "old-lock"}), encoding="utf-8"
    )
    (unrelated / "keep.txt").write_text("not ours", encoding="utf-8")
    env = _worker_env(tmp_path / "project")
    monkeypatch.setattr(_environment, "engine_cache_root", lambda: cache)
    monkeypatch.setattr(_environment, "lock_hash", lambda _: "current-lock")

    removed = _environment.prune_prepared({"demo": env})

    assert removed == [stale]
    assert current.is_dir()
    assert not stale.exists()
    assert unrelated.is_dir()


def test_list_prepared_reports_managed_env_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    prepared = cache / "demo-cpu-abcd"
    venv = prepared / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_bytes(b"python")
    marker = {
        "framework": "demo",
        "accelerator": "cpu",
        "lock_hash": "abcd1234",
        "distribution": "demo-plugin",
        "format_version": 1,
    }
    (prepared / ".physicalai-engine.json").write_text(json.dumps(marker), encoding="utf-8")
    monkeypatch.setattr(_environment, "engine_cache_root", lambda: cache)

    environments = _environment.list_prepared()

    assert len(environments) == 1
    assert environments[0].framework == "demo"
    assert environments[0].accelerator == "cpu"
    assert environments[0].lock_hash == "abcd1234"
    assert environments[0].size_bytes >= len(b"python")
    assert environments[0].path == prepared
