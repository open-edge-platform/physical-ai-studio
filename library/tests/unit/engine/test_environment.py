# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine metadata and Runtime isolate adapter tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from physicalai import isolate
from physicalai.engine import ProjectSource, WorkerEnv, to_isolate_environment
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


def test_to_isolate_environment_maps_worker_python_and_accelerator_extras(tmp_path: Path) -> None:
    env = _generated_env(
        tmp_path,
        accelerators=("cpu", "cuda"),
        python="3.12",
        extras={"cpu": (), "cu128": ("cuda",)},
    )

    environment = to_isolate_environment(env, "cuda")

    assert isinstance(environment, isolate.Environment)
    assert environment.source is env.source
    assert environment.profile == isolate.SyncProfile(python="3.12", extras=("cuda",))


def test_prepare_delegates_to_runtime_isolate_and_keeps_engine_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = _generated_env(tmp_path, python="3.12", extras={"cpu": ()})
    cache = tmp_path / "cache"
    monkeypatch.setattr(_environment, "engine_cache_root", lambda: cache)
    calls: list[tuple[isolate.Environment, isolate.UvProvider, bool]] = []

    def fake_prepare(
        environment: isolate.Environment,
        *,
        provider: isolate.UvProvider,
        offline: bool,
    ) -> isolate.PreparedEnvironment:
        calls.append((environment, provider, offline))
        root = cache / environment.key()
        project = root / "project"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname='gr00t'\n", encoding="utf-8")
        (project / "uv.lock").write_text(
            'version = 1\n\n[[package]]\nname = "gr00t"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        python = root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("python", encoding="utf-8")
        return isolate.PreparedEnvironment(path=root, key=environment.key(), profile=environment.profile)

    monkeypatch.setattr(isolate, "prepare", fake_prepare)

    prepared = _environment.prepare_environment(env, accelerator="auto", offline=True)

    assert prepared.framework == "gr00t"
    assert prepared.name == "openvino"
    assert prepared.accelerator == "cpu"
    assert prepared.key == env.key()
    assert prepared.distribution == env.distribution
    assert prepared.python == prepared.path / "venv" / "bin" / "python"
    assert prepared.isolate.key == calls[0][0].key()
    assert calls[0][0].source is env.source
    assert calls[0][0].profile == isolate.SyncProfile(python="3.12", extras=())
    assert calls[0][1].cache_dir == cache
    assert calls[0][2] is True
    marker = json.loads((prepared.path / ".physicalai-engine.json").read_text(encoding="utf-8"))
    assert marker["source"] == {"kind": "generated", "revision": "a" * 40}
    assert marker["isolate_key"] == calls[0][0].key()
    assert str(_environment.locked_version(env, prepared.project)) == "0.1.0"
    with pytest.raises(ValueError, match="only after preparation"):
        _environment.locked_version(env)


def test_accelerator_auto_prefers_xpu_then_cuda_then_cpu_independent_of_host_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"cuda"})
    assert _environment.resolve_accelerator("auto") == "cu128"

    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"xpu", "cuda"})
    assert _environment.resolve_accelerator("auto") == "xpu"

    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": True})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"xpu"})
    assert _environment.resolve_accelerator("auto") == "cu128"

    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: set())
    assert _environment.resolve_accelerator("auto") == "cpu"

    with pytest.raises(ValueError, match="Unsupported accelerator"):
        _environment.resolve_accelerator("mps")


def test_hardware_detection_reads_pci_vendor_and_display_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def add_device(name: str, *, vendor: str, class_code: str) -> None:
        device = tmp_path / name
        device.mkdir()
        (device / "vendor").write_text(vendor, encoding="ascii")
        (device / "class").write_text(class_code, encoding="ascii")

    add_device("nvidia", vendor="0x10de", class_code="0x030000")
    add_device("intel-display", vendor="0x8086", class_code="0x038000")
    add_device("intel-audio", vendor="0x8086", class_code="0x040000")
    add_device("amd-display", vendor="0x1002", class_code="0x030000")
    monkeypatch.setattr(_environment, "_command_reports_gpu", Mock(return_value=False))

    assert _environment._hardware_accelerators(tmp_path) == {"cuda", "xpu"}


def test_explicit_cpu_override_is_honored_when_gpu_hardware_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"cuda"})

    assert _environment.resolve_accelerator("cpu") == "cpu"


def test_auto_rejects_gpu_hardware_that_engine_environment_cannot_provide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"cuda"})

    with pytest.raises(ValueError, match="Detected CUDA hardware.*supports"):
        _environment.resolve_accelerator("auto", allowed=("cpu", "xpu"))


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


def _marker(env: WorkerEnv, **overrides: object) -> str:
    marker = {
        "format_version": 3,
        "framework": env.framework,
        "name": env.name,
        "accelerator": "cpu",
        "key": env.key(),
        "distribution": env.distribution,
        "source": env.source.identity(),
        "isolate_key": "1" * 64,
    }
    marker.update(overrides)
    return json.dumps(marker)


def test_prune_removes_only_stale_engine_managed_environments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    env = _worker_env(tmp_path / "project")
    enablement_env = WorkerEnv(
        framework="demo",
        name="openvino",
        project=env.project,
        worker=env.worker,
        distribution="demo-plugin",
        package="lerobot",
        python="3.12",
    )
    directories = {
        name: cache / name for name in ("current", "enablement", "stale", "removed-name", "legacy", "unrelated")
    }
    for directory in directories.values():
        directory.mkdir()
    (directories["current"] / ".physicalai-engine.json").write_text(_marker(env), encoding="utf-8")
    (directories["enablement"] / ".physicalai-engine.json").write_text(_marker(enablement_env), encoding="utf-8")
    (directories["stale"] / ".physicalai-engine.json").write_text(_marker(env, key="0" * 64), encoding="utf-8")
    (directories["removed-name"] / ".physicalai-engine.json").write_text(_marker(env, name="gone"), encoding="utf-8")
    (directories["legacy"] / ".physicalai-engine.json").write_text(
        json.dumps({"format_version": 2, "framework": "demo", "lock_hash": "old"}),
        encoding="utf-8",
    )
    (directories["unrelated"] / "keep.txt").write_text("not ours", encoding="utf-8")
    monkeypatch.setattr(_environment, "engine_cache_root", lambda: cache)

    removed = _environment.prune_prepared([env, enablement_env])

    assert removed == sorted([directories["legacy"], directories["removed-name"], directories["stale"]])
    assert directories["current"].is_dir()
    assert directories["enablement"].is_dir()
    assert directories["unrelated"].is_dir()


def test_list_prepared_reports_engine_metadata_for_isolate_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    env = _worker_env(tmp_path / "project")
    prepared = cache / ("1" * 64)
    venv = prepared / "venv" / "bin"
    venv.mkdir(parents=True)
    (prepared / "project").mkdir()
    (venv / "python").write_bytes(b"python")
    (prepared / ".physicalai-isolate.json").write_text("{}\n", encoding="utf-8")
    (prepared / ".physicalai-engine.json").write_text(_marker(env), encoding="utf-8")
    monkeypatch.setattr(_environment, "engine_cache_root", lambda: cache)

    environments = _environment.list_prepared()

    assert len(environments) == 1
    assert environments[0].framework == "demo"
    assert environments[0].name == "default"
    assert environments[0].accelerator == "cpu"
    assert environments[0].key == env.key()
    assert environments[0].size_bytes >= len(b"python")
    assert environments[0].path == prepared
    assert environments[0].python == prepared / "venv" / "bin" / "python"


def test_engine_cache_defaults_to_runtime_isolate_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("PHYSICALAI_ENGINE_CACHE", raising=False)
    monkeypatch.setenv("PHYSICALAI_ISOLATE_CACHE", str(tmp_path / "isolate"))

    assert _environment.engine_cache_root() == (tmp_path / "isolate").resolve()


def test_worker_environment_keys_change_with_every_input(tmp_path: Path) -> None:
    env = _generated_env(tmp_path)
    variants = [
        _generated_env(tmp_path, project=_GeneratedProject("b" * 40)),
        _generated_env(tmp_path, name="native"),
        _generated_env(tmp_path, python="3.10"),
        _generated_env(tmp_path, extras={"cpu": ("cpu",)}),
        _generated_env(tmp_path, framework="other"),
    ]
    assert env.key() == _generated_env(tmp_path).key()
    assert len({env.key(), *(variant.key() for variant in variants)}) == len(variants) + 1


def test_single_profile_environments_skip_hardware_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": True})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"cuda"})

    assert _environment.resolve_accelerator("auto", allowed=("cpu",)) == "cpu"
    with pytest.raises(ValueError, match="not supported"):
        _environment.resolve_accelerator("xpu", allowed=("cpu",))


def test_plugins_for_git_pinned_frameworks_declare_no_version_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(_environment.metadata, "requires", lambda distribution: ["physicalai~=0.3.0"])

    assert _generated_env(tmp_path).supported() is None


class _GeneratedProject(ProjectSource):
    """A source whose lock exists only after materialization, like a Git checkout."""

    def __init__(self, revision: str) -> None:
        self.revision = revision

    def identity(self) -> dict[str, object]:
        return {"kind": "generated", "revision": self.revision}

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        del offline
        destination.mkdir(parents=True)
        (destination / "pyproject.toml").write_text("[project]\nname='gr00t'\n", encoding="utf-8")
        (destination / "uv.lock").write_text(
            'version = 1\n\n[[package]]\nname = "gr00t"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )


def _generated_env(tmp_path: Path, **overrides: object) -> WorkerEnv:
    values: dict[str, object] = {
        "framework": "gr00t",
        "project": _GeneratedProject("a" * 40),
        "worker": tmp_path / "worker.py",
        "distribution": "demo-plugin",
        "package": "gr00t",
        "accelerators": ("cpu",),
        "name": "openvino",
        "python": "3.12",
        "extras": {"cpu": ()},
    }
    values.update(overrides)
    return WorkerEnv(**values)  # type: ignore[arg-type]
