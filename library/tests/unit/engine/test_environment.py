# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine metadata and Runtime isolate adapter tests."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
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


def test_each_environment_chooses_the_extra_for_its_cuda_profile(tmp_path: Path) -> None:
    assert _environment.resolve_accelerator("cu128") == "cuda"
    assert _environment.resolve_accelerator("cu132") == "cuda"
    assert _environment.resolve_accelerator("cu132", allowed=("cpu", "cu128")) == "cuda"

    mapped = _generated_env(
        tmp_path / "mapped",
        accelerators=("cpu", "cuda"),
        extras={"cpu": ("cpu",), "cuda": ("cu132",)},
    )
    legacy = _generated_env(tmp_path / "legacy", accelerators=("cu128",), extras=None)
    default = _generated_env(tmp_path / "default", accelerators=("cpu", "cuda", "xpu"), extras=None)

    assert mapped.extras_for("cuda") == ("cu132",)
    assert mapped.extras_for("cu128") == ("cu132",)
    assert legacy.extras_for("cuda") == ("cu128",)
    assert default.extras_for("cuda") == ("cuda",)
    assert to_isolate_environment(mapped, "cuda").profile.extras == ("cu132",)


def test_prepare_delegates_to_runtime_isolate_and_keeps_engine_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = _generated_env(tmp_path, python="3.12", extras={"cpu": ()})
    cache = tmp_path / "cache"
    calls: list[tuple[isolate.Environment, isolate.UvProvider, bool, Mapping[str, str]]] = []

    def fake_prepare(
        environment: isolate.Environment,
        *,
        provider: isolate.UvProvider,
        offline: bool,
        labels: Mapping[str, str],
    ) -> isolate.PreparedEnvironment:
        calls.append((environment, provider, offline, labels))
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
        return isolate.PreparedEnvironment(
            path=root,
            key=environment.key(),
            profile=environment.profile,
            labels=labels,
        )

    monkeypatch.setattr(isolate, "prepare", fake_prepare)
    monkeypatch.setattr("uv.find_uv_bin", lambda: "/venv/bin/uv")

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
    # Engine shares isolate's cache (PHYSICALAI_ISOLATE_CACHE) and records its metadata as labels.
    assert calls[0][1].cache_dir is None
    assert calls[0][1].uv == "/venv/bin/uv"
    assert calls[0][2] is True
    assert calls[0][3] == _engine_labels(env, name="openvino")
    assert str(_environment.locked_version(env, prepared.project)) == "0.1.0"
    with pytest.raises(ValueError, match="only after preparation"):
        _environment.locked_version(env)


def test_accelerator_auto_prefers_xpu_then_cuda_then_cpu_independent_of_host_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": False})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"cuda"})
    assert _environment.resolve_accelerator("auto") == "cuda"

    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"xpu", "cuda"})
    assert _environment.resolve_accelerator("auto") == "xpu"

    monkeypatch.setattr(_environment, "_torch_accelerators", lambda: {"xpu": False, "cuda": True})
    monkeypatch.setattr(_environment, "_hardware_accelerators", lambda: {"xpu"})
    assert _environment.resolve_accelerator("auto") == "cuda"

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


@pytest.mark.skipif(os.name != "posix", reason="creating symbolic links may need privileges on Windows")
def test_bundled_worker_lock_must_be_a_regular_file(tmp_path: Path) -> None:
    env = _worker_env(tmp_path / "project")
    lock = tmp_path / "project" / "uv.lock"
    real_lock = tmp_path / "elsewhere.lock"
    lock.rename(real_lock)
    lock.symlink_to(real_lock)

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        _environment._lock_bytes(env)

    lock.unlink()
    with pytest.raises(FileNotFoundError, match="does not include uv.lock"):
        _environment._lock_bytes(env)


def _engine_labels(env: WorkerEnv, **overrides: str) -> dict[str, str]:
    metadata = {
        "framework": env.framework,
        "name": env.name,
        "accelerator": "cpu",
        "key": env.key(),
        "distribution": env.distribution,
        **overrides,
    }
    return {f"engine.{name}": value for name, value in metadata.items()}


def _isolate_entry(root: Path, labels: Mapping[str, str]) -> isolate.PreparedEnvironment:
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    (root / "project").mkdir()
    return isolate.PreparedEnvironment(path=root, key=root.name, profile=isolate.SyncProfile(), labels=labels)


def test_prune_removes_only_stale_engine_managed_environments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
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
    entries = {
        "current": _isolate_entry(cache / ("1" * 64), _engine_labels(env)),
        "enablement": _isolate_entry(cache / ("2" * 64), _engine_labels(enablement_env)),
        "stale": _isolate_entry(cache / ("3" * 64), _engine_labels(env, key="0" * 64)),
        "removed-name": _isolate_entry(cache / ("4" * 64), _engine_labels(env, name="gone")),
        # Prepared by another application in the shared cache, or labelled only in part.
        "runtime": _isolate_entry(cache / ("5" * 64), {"owner": "runtime"}),
        "partial": _isolate_entry(cache / ("6" * 64), {"engine.framework": "demo"}),
    }
    removed_entries: list[isolate.PreparedEnvironment] = []
    monkeypatch.setattr(isolate, "list_prepared", lambda: list(entries.values()))
    monkeypatch.setattr(isolate, "remove_prepared", removed_entries.append)

    removed = _environment.prune_prepared([env, enablement_env])

    assert removed == sorted([entries["stale"].path, entries["removed-name"].path])
    assert sorted(entry.path for entry in removed_entries) == removed


def test_prune_reads_labels_without_measuring_environments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Measuring walks every file of an environment, and pruning needs only its labels."""
    from physicalai.engine import Engine  # ruff: ignore[import-outside-top-level]

    cache = tmp_path / "cache"
    env = _worker_env(tmp_path / "project")
    current = _isolate_entry(cache / ("1" * 64), _engine_labels(env))
    stale = _isolate_entry(cache / ("2" * 64), _engine_labels(env, key="0" * 64))
    removed_entries: list[isolate.PreparedEnvironment] = []
    monkeypatch.setattr(isolate, "list_prepared", lambda: [current, stale])
    monkeypatch.setattr(isolate, "remove_prepared", removed_entries.append)
    monkeypatch.setattr(Engine, "get", Mock(return_value=Mock(environments=Mock(return_value=[env]))))
    monkeypatch.setattr(_environment, "_directory_size", Mock(side_effect=AssertionError("measured an environment")))

    assert _environment.prune() == [stale.path]
    assert removed_entries == [stale]


def test_list_prepared_reports_engine_metadata_for_isolate_cache_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    env = _worker_env(tmp_path / "project")
    prepared = _isolate_entry(cache / ("1" * 64), _engine_labels(env))
    runtime_only = _isolate_entry(cache / ("2" * 64), {})
    monkeypatch.setattr(isolate, "list_prepared", lambda: [runtime_only, prepared])

    environments = _environment.list_prepared()

    assert len(environments) == 1
    assert environments[0].framework == "demo"
    assert environments[0].name == "default"
    assert environments[0].accelerator == "cpu"
    assert environments[0].key == env.key()
    assert environments[0].distribution == "demo-plugin"
    assert environments[0].size_bytes >= len(b"python")
    assert environments[0].isolate is prepared
    assert environments[0].path == prepared.path
    assert environments[0].python == prepared.path / "venv" / "bin" / "python"


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


def test_environment_keys_are_pinned_for_a_fixed_project(tmp_path: Path) -> None:
    """Pin both keys so a change to either is deliberate.

    The isolate key names the cache directory, so changing it makes every user rebuild their
    environments. The Engine key decides what ``physicalai engine prune`` treats as stale.
    Both depend only on file contents, so they are the same on every machine.
    """
    env = _worker_env(tmp_path / "project")

    assert env.key() == "64bde4ef6e3934d58898d6b5d772008fccaae333559a18000550c617abfe27b2"
    assert {accelerator: to_isolate_environment(env, accelerator).key() for accelerator in ("cpu", "cuda", "xpu")} == {
        "cpu": "66857cfe1ed66a1904e0c8977a5261c3f05885444ff35f5ac780c4ab4bdaff47",
        "cuda": "f27a1f30d1e0fb19607614b68dd828a570ad22a9b941017c387eebc1c6d18e43",
        "xpu": "9c4e643abadf5e651179f0843b64b76fec36c6b50e83139c16adacae291e86fd",
    }


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


def test_installed_uv_falls_back_to_path_when_the_uv_package_has_no_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> str:
        raise FileNotFoundError

    monkeypatch.setattr("uv.find_uv_bin", missing)

    assert _environment._installed_uv() is None
