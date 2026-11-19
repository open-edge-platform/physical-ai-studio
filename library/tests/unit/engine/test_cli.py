# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native training and engine-management CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from jsonargparse import ArgumentParser, Namespace

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

    _FakeEngine = type("FakeEngine", (), {"env": env, "framework": "lerobot"})

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
