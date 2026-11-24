# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native training and engine-management CLI tests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from jsonargparse import ArgumentParser, Namespace
from physicalai import isolate
from uv import find_uv_bin

from physicalai.cli import engine as engine_cli
from physicalai.cli import train as train_cli
from physicalai.engine import Engine, TrainingEngine, WorkerEnv


def test_train_parser_keeps_native_device_arguments_opaque() -> None:
    cfg = train_cli.build_parser().parse_args(
        [
            "--interpreter",
            "/opt/lerobot/bin/python",
            "lerobot",
            "--help",
            "--policy.device=cuda",
            "--accelerator",
            "native-value",
        ],
    )

    assert cfg.framework == "lerobot"
    assert cfg.interpreter == Path("/opt/lerobot/bin/python")
    assert not hasattr(cfg, "accelerator")
    assert cfg.args == ["--help", "--policy.device=cuda", "--accelerator", "native-value"]


def test_train_dispatch_returns_the_framework_native_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[tuple[str, ...], str | Path | None]] = []

    class _FakeEngine(TrainingEngine):
        framework = "lerobot"

        @classmethod
        def cli(cls, args: list[str], *, interpreter: str | Path | None = None) -> int:  # type: ignore[override]
            events.append((tuple(args), interpreter))
            return 23

    monkeypatch.setattr(Engine, "get", classmethod(lambda cls, name: _FakeEngine))
    parser = train_cli.build_parser()
    cfg = parser.parse_args(
        ["--interpreter", "python-custom", "lerobot", "--policy.device", "cuda", "--steps=2"],
    )

    assert train_cli.run(parser, cfg) == 23
    assert events == [( ("--policy.device", "cuda", "--steps=2"), Path("python-custom"))]


def test_engine_prepare_uses_the_registered_engines_locked_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env = WorkerEnv(
        framework="lerobot",
        project=tmp_path,
        worker=tmp_path / "worker.py",
        distribution="demo-plugin",
        package="lerobot",
    )

    _FakeEngine = type("FakeEngine", (Engine,), {"env": env, "framework": "lerobot"})

    calls: list[tuple[object, str, bool]] = []
    monkeypatch.setattr(Engine, "get", classmethod(lambda cls, name: _FakeEngine))

    def fake_prepare(current: WorkerEnv, accelerator: str, *, offline: bool) -> Path:
        calls.append((current, accelerator, offline))
        return tmp_path / "python"

    monkeypatch.setattr(engine_cli, "prepare", fake_prepare)
    parser = engine_cli.build_parser()
    cfg = parser.parse_args(["prepare", "lerobot", "--accelerator", "xpu", "--offline"])

    assert engine_cli.run(parser, cfg) == 0
    assert calls == [(env, "xpu", True)]


def test_engine_list_and_prune_use_core_environment_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = type(
        "PreparedEnvironment",
        (),
        {
            "framework": "lerobot",
            "name": "default",
            "accelerator": "cpu",
            "key": "abcd12345678",
            "size_bytes": 1024,
            "path": Path("/cache/lerobot"),
        },
    )()
    removed = [Path("/cache/old")]
    monkeypatch.setattr(engine_cli, "list_prepared", lambda: [listed])
    monkeypatch.setattr(engine_cli, "prune", lambda: removed)
    outputs: list[str] = []
    monkeypatch.setattr(engine_cli, "print", lambda value: outputs.append(str(value)), raising=False)
    parser = engine_cli.build_parser()

    assert engine_cli.run(parser, parser.parse_args(["list"])) == 0
    assert engine_cli.run(parser, parser.parse_args(["prune"])) == 0
    assert any("old" in line for line in outputs)


def test_train_rejects_engines_that_only_package_models(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExportOnly(Engine):
        framework = "export-only"

    monkeypatch.setattr(Engine, "get", classmethod(lambda cls, name: _ExportOnly))
    parser = train_cli.build_parser()

    with pytest.raises(ValueError, match="does not train"):
        train_cli.run(parser, parser.parse_args(["export-only", "--help"]))


def test_engine_prepare_selects_an_enablement_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from physicalai.engine import Enablement

    base = WorkerEnv(framework="demo", project=tmp_path, worker=tmp_path / "w.py", distribution="d", package="p")
    patched = WorkerEnv(
        framework="demo",
        name="openvino",
        project=tmp_path,
        worker=tmp_path / "w.py",
        distribution="d",
        package="p",
    )

    class _Demo(Engine):
        framework = "demo"
        env = base
        enablements = (Enablement(name="openvino", env=patched),)

    calls: list[WorkerEnv] = []

    def prepare(env: WorkerEnv, accelerator: str, *, offline: bool) -> Path:
        del accelerator, offline
        calls.append(env)
        return tmp_path

    monkeypatch.setattr(Engine, "get", classmethod(lambda cls, name: _Demo))
    monkeypatch.setattr(engine_cli, "prepare", prepare)
    parser = engine_cli.build_parser()

    assert engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--enablement", "openvino"])) == 0
    assert calls == [patched]
    with pytest.raises(ValueError, match="no enablement environment 'missing'"):
        engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--enablement", "missing"]))


def test_engine_prepare_selects_a_policy_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = WorkerEnv(framework="demo", project=tmp_path, worker=tmp_path / "w.py", distribution="d", package="p")
    smol = WorkerEnv(
        framework="demo",
        name="smol",
        project=tmp_path / "smol",
        worker=tmp_path / "w.py",
        distribution="d",
        package="p",
    )

    class _Demo(Engine):
        framework = "demo"
        env = base
        policy_envs = {"smolvla": smol}  # noqa: RUF012

        @classmethod
        def policies(cls) -> list[str]:
            return ["act", "smolvla"]

    calls: list[WorkerEnv] = []

    def prepare(env: WorkerEnv, accelerator: str, *, offline: bool) -> Path:
        del accelerator, offline
        calls.append(env)
        return tmp_path

    monkeypatch.setattr(Engine, "get", classmethod(lambda cls, name: _Demo))
    monkeypatch.setattr(engine_cli, "prepare", prepare)
    parser = engine_cli.build_parser()

    assert engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--policy", "smolvla"])) == 0
    assert engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--policy", "act"])) == 0
    assert calls == [smol, base]
    with pytest.raises(ValueError, match="either --enablement or --policy"):
        engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--policy", "smolvla", "--enablement", "x"]))
    with pytest.raises(ValueError, match=r"demo has no policy 'smol_vla'; choose one of: act, smolvla"):
        engine_cli.run(parser, parser.parse_args(["prepare", "demo", "--policy", "smol_vla"]))
    assert calls == [smol, base]


_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _empty_worker_project(root: Path, name: str) -> Path:
    """Write a dependency-free locked uv project that prepares offline in well under a second."""
    project = root / name
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">={_PYTHON}"\n',
        encoding="utf-8",
    )
    (project / "uv.lock").write_text(
        f'version = 1\nrevision = 3\nrequires-python = ">={_PYTHON}"\n\n'
        f'[[package]]\nname = "{name}"\nversion = "0.1.0"\nsource = {{ virtual = "." }}\n',
        encoding="utf-8",
    )
    return project


def _cpu_engine(framework: str, project: Path) -> type[Engine]:
    env = WorkerEnv(
        framework=framework,
        project=project,
        worker=project / "pyproject.toml",
        distribution=f"{framework}-plugin",
        package=project.name,
        accelerators=("cpu",),
        extras={"cpu": ()},
        python=_PYTHON,
    )
    return type(f"_{framework.title()}Engine", (Engine,), {"framework": framework, "env": env})


def test_engine_prepare_list_and_prune_manage_real_worker_environments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Characterize the user-facing environment lifecycle end to end, through the CLI only.

    Environments are prepared for real (offline, no dependencies) into a temporary cache, so this
    test does not depend on how Engine records or finds them and must keep passing when that
    bookkeeping moves into physicalai.isolate.
    """
    cache = tmp_path / "cache"
    monkeypatch.delenv("PHYSICALAI_ENGINE_CACHE", raising=False)
    monkeypatch.setenv("PHYSICALAI_ISOLATE_CACHE", str(cache))
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    engines = {
        "demo": _cpu_engine("demo", _empty_worker_project(tmp_path / "v1", "demo-worker")),
        "gone": _cpu_engine("gone", _empty_worker_project(tmp_path / "v1", "gone-worker")),
    }

    def get(cls: type[Engine], name: str) -> type[Engine]:
        del cls
        if name not in engines:
            msg = f"No engine named {name!r}."
            raise ValueError(msg)
        return engines[name]

    monkeypatch.setattr(Engine, "get", classmethod(get))
    parser = engine_cli.build_parser()

    def cli(*arguments: str) -> list[str]:
        assert engine_cli.run(parser, parser.parse_args(list(arguments))) == 0
        return capsys.readouterr().out.splitlines()

    assert cli("list") == ["No prepared worker environments."]
    prepared = {}
    for framework in engines:
        (line,) = cli("prepare", framework, "--accelerator", "cpu", "--offline")
        prefix = f"Prepared {framework} default worker environment: "
        assert line.startswith(prefix)
        python = Path(line.removeprefix(prefix))
        assert python.is_file()
        assert python.is_relative_to(cache.resolve())
        prepared[framework] = python.parents[2]
    assert prepared["demo"] != prepared["gone"]

    # Runtime shares the cache; Engine must neither list nor prune environments it did not prepare.
    runtime_only = isolate.prepare(
        isolate.Environment(
            isolate.BundledProject(_empty_worker_project(tmp_path, "runtime-only")),
            isolate.SyncProfile(python=_PYTHON),
        ),
        provider=isolate.UvProvider(uv=find_uv_bin()),
        offline=True,
    )
    assert runtime_only.path.is_relative_to(cache.resolve())

    rows = [line.split("\t") for line in cli("list")]
    assert [row[:3] for row in rows] == [["demo", "default", "cpu"], ["gone", "default", "cpu"]]
    for row, framework in zip(rows, ("demo", "gone"), strict=True):
        assert row[3] == engines[framework].env.key()[:12]
        assert re.fullmatch(r"\d+\.\d (B|KiB|MiB|GiB)", row[4])
        assert Path(row[5]) == prepared[framework]

    assert cli("prune") == ["No stale worker environments."]

    # The demo plugin moves to a new lock and the gone plugin is uninstalled: both are now stale.
    engines["demo"] = _cpu_engine("demo", _empty_worker_project(tmp_path / "v2", "demo-worker-v2"))
    del engines["gone"]
    assert sorted(cli("prune")) == sorted(
        f"Removed stale worker environment: {prepared[framework]}" for framework in ("demo", "gone")
    )
    assert not prepared["demo"].exists()
    assert not prepared["gone"].exists()
    assert runtime_only.python.is_file()
    assert cli("list") == ["No prepared worker environments."]
    assert cli("prune") == ["No stale worker environments."]
