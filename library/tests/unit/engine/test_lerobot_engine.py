# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Public LeRobot engine contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

from physicalai.engine import Engine, Run, TrainHooks
from physicalai_lerobot_policy_plugin import engine as lerobot_engine_module
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine, settings_to_flags


def test_lerobot_engine_is_discovered_as_an_engine_subclass() -> None:
    assert issubclass(LeRobotEngine, Engine)
    assert Engine.get("lerobot") is LeRobotEngine
    assert LeRobotEngine.framework == "lerobot"
    assert LeRobotEngine.env is not None
    assert LeRobotEngine.env.framework == "lerobot"


def test_lerobot_schema_and_policy_catalog_come_from_native_config_types() -> None:
    """LeRobot policies and schemas are derived from its config registry."""
    policies = LeRobotEngine.policies()
    schema = LeRobotEngine.config_schema("act")

    assert "act" in policies
    assert "diffusion" in policies
    assert "groot" not in policies
    assert schema["title"] == "TrainPipelineConfig"
    assert schema["properties"]["policy"]["title"] == "ACTConfig"


def test_lerobot_cli_uses_native_device_for_worker_extra_and_passes_args_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []
    prepared: list[str] = []

    def fake_interpreter_for(env: object, *, accelerator: str, interpreter: str | Path | None) -> Path:
        del env, interpreter
        prepared.append(accelerator)
        return Path("/env/bin/python")

    def fake_call(command: list[str], *, shell: bool) -> int:
        calls.append((command, shell))
        return 2

    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", fake_interpreter_for)
    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", fake_call)

    argv = ["--help", "--policy.device", "cuda", "--policy.type=act"]
    result = LeRobotEngine.cli(argv)

    assert result == 2
    assert prepared == ["cu128"]
    assert calls == [(["/env/bin/lerobot-train", *argv], False)]


def test_lerobot_cli_auto_selects_worker_extra_when_device_is_unspecified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str] = []

    def fake_resolve(accelerator: str, *, allowed: tuple[str, ...]) -> str:
        selected.append(accelerator)
        assert "cu128" in allowed
        return "cu128"

    def fake_interpreter_for(env: object, *, accelerator: str, interpreter: str | Path | None) -> Path:
        del env, interpreter
        assert accelerator == "cu128"
        return Path("/env/bin/python")

    monkeypatch.setattr(lerobot_engine_module, "resolve_accelerator", fake_resolve)
    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", fake_interpreter_for)
    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", lambda command, *, shell: 0)

    assert LeRobotEngine.cli(["--steps=2"]) == 0
    assert selected == ["auto"]


def test_lerobot_cli_uses_native_config_device_to_select_worker_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "train_config.json"
    config.write_text('{"policy":{"device":"xpu"}}', encoding="utf-8")
    prepared: list[str] = []
    def fake_interpreter_for(env: object, *, accelerator: str, interpreter: str | Path | None) -> Path:
        del env, interpreter
        prepared.append(accelerator)
        return Path("/env/bin/python")

    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", fake_interpreter_for)
    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", lambda command, *, shell: 0)

    assert LeRobotEngine.cli(["--config_path", str(config)]) == 0
    assert prepared == ["xpu"]


def test_lerobot_cli_uses_policy_path_config_to_select_worker_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "pretrained_model"
    policy_path.mkdir()
    (policy_path / "config.json").write_text('{"device":"xpu"}', encoding="utf-8")
    prepared: list[str] = []

    def fake_interpreter_for(env: object, *, accelerator: str, interpreter: str | Path | None) -> Path:
        del env, interpreter
        prepared.append(accelerator)
        return Path("/env/bin/python")

    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", fake_interpreter_for)
    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", lambda command, *, shell: 0)

    assert LeRobotEngine.cli(["--policy.path", str(policy_path)]) == 0
    assert prepared == ["xpu"]


def test_lerobot_cli_native_flag_overrides_device_in_config_without_rewriting_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "train_config.json"
    config.write_text('{"policy":{"device":"xpu"}}', encoding="utf-8")
    prepared: list[str] = []
    calls: list[list[str]] = []

    def fake_interpreter_for(env: object, *, accelerator: str, interpreter: str | Path | None) -> Path:
        del env, interpreter
        prepared.append(accelerator)
        return Path("/env/bin/python")

    def fake_call(command: list[str], *, shell: bool) -> int:
        del shell
        calls.append(command)
        return 0

    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", fake_interpreter_for)
    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", fake_call)
    argv = [f"--config_path={config}", "--policy.device=cuda"]

    assert LeRobotEngine.cli(argv) == 0
    assert prepared == ["cu128"]
    assert calls == [["/env/bin/lerobot-train", *argv]]


def test_lerobot_constructor_no_longer_accepts_worker_accelerator() -> None:
    with pytest.raises(TypeError, match="accelerator"):
        LeRobotEngine(accelerator="xpu")


def test_lerobot_train_serializes_native_config_and_checks_exact_worker_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "train_config.json"
    events: list[tuple[object, ...]] = []
    expected_run = Run(framework="lerobot", output_dir=tmp_path / "run", checkpoint=None)

    class _Policy:
        device = "xpu"

    class _Config:
        policy = _Policy()
        steps = 4

        def save_pretrained(self, directory: Path) -> None:
            (directory / "train_config.json").write_text("{}", encoding="utf-8")
            events.append(("saved", directory))

    def fake_run_worker(
        env: object,
        args: list[str],
        *,
        device: str,
        hooks: TrainHooks | None,
        accelerator: str,
        interpreter: Path | None,
        expected_version: Version,
    ) -> Run:
        events.append(("run", env, args, device, hooks, accelerator, interpreter, expected_version))
        assert Path(args[0].partition("=")[2]).is_file()
        return expected_run

    hooks = object()
    monkeypatch.setattr(lerobot_engine_module, "_host_lerobot_version", lambda env: Version("0.6.1"))
    monkeypatch.setattr(lerobot_engine_module, "run_worker", fake_run_worker)

    result = LeRobotEngine().train(_Config(), hooks=hooks)  # type: ignore[arg-type]

    assert result is expected_run
    assert events[0][0] == "saved"
    assert events[1][0] == "run"
    assert events[1][3:] == ("xpu", hooks, "xpu", None, Version("0.6.1"))
    assert not config_file.exists()


def test_lerobot_train_auto_selects_a_matching_worker_for_unspecified_policy_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []

    class _Policy:
        device = None

    class _Config:
        policy = _Policy()

        def save_pretrained(self, directory: Path) -> None:
            (directory / "train_config.json").write_text("{}", encoding="utf-8")

    def fake_run_worker(
        env: object,
        args: list[str],
        *,
        device: str,
        hooks: TrainHooks | None,
        accelerator: str,
        interpreter: Path | None,
        expected_version: Version,
    ) -> Run:
        del env, args, hooks, interpreter, expected_version
        events.append((device, accelerator))
        return Run(framework="lerobot", output_dir=tmp_path / "run", checkpoint=None)

    def fake_resolve(requested: str, *, allowed: tuple[str, ...]) -> str:
        assert requested == "auto"
        assert "cu128" in allowed
        return "cu128"

    monkeypatch.setattr(lerobot_engine_module, "_host_lerobot_version", lambda env: Version("0.6.1"))
    monkeypatch.setattr(lerobot_engine_module, "resolve_accelerator", fake_resolve)
    monkeypatch.setattr(lerobot_engine_module, "run_worker", fake_run_worker)

    LeRobotEngine().train(_Config())  # type: ignore[arg-type]

    assert events == [("cuda", "cu128")]


def test_lerobot_run_overrides_dataset_output_and_device_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    expected_run = Run(framework="lerobot", output_dir=tmp_path / "run", checkpoint=None)

    def fake_run_worker(env: object, args: list[str], **kwargs: object) -> Run:
        del env
        calls.append((args, kwargs))
        return expected_run

    monkeypatch.setattr(lerobot_engine_module, "run_worker", fake_run_worker)
    settings = {
        "policy": {"type": "act", "device": "cuda"},
        "dataset": {"repo_id": "hub/repo", "root": "wrong"},
        "steps": 8,
    }

    result = LeRobotEngine.run(
        settings,
        dataset=tmp_path / "snapshot",
        output_dir=tmp_path / "output",
        device="cpu",
        interpreter=Path("/custom/python"),
    )

    assert result is expected_run
    args = calls[0][0]
    assert "--policy.type=act" in args
    assert "--steps=8" in args
    assert "--dataset.repo_id=local" in args
    assert f"--dataset.root={tmp_path / 'snapshot'}" in args
    assert f"--output_dir={tmp_path / 'output'}" in args
    assert "--policy.device=cpu" in args
    assert not any("hub/repo" in value or "wrong" in value or "cuda" in value for value in args)
    assert calls[0][1]["accelerator"] == "cpu"
    assert calls[0][1]["interpreter"] == Path("/custom/python")


def test_lerobot_run_auto_selects_device_and_matching_worker_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    expected_run = Run(framework="lerobot", output_dir=tmp_path / "run", checkpoint=None)

    def fake_run_worker(env: object, args: list[str], **kwargs: object) -> Run:
        del env
        calls.append((args, kwargs))
        return expected_run

    monkeypatch.setattr(lerobot_engine_module, "resolve_accelerator", lambda *_args, **_kwargs: "cu128")
    monkeypatch.setattr(lerobot_engine_module, "run_worker", fake_run_worker)

    result = LeRobotEngine.run(
        {"policy": {"type": "act"}},
        dataset=tmp_path / "snapshot",
        output_dir=tmp_path / "output",
        device="auto",
    )

    assert result is expected_run
    assert "--policy.device=cuda" in calls[0][0]
    assert calls[0][1]["device"] == "cuda"
    assert calls[0][1]["accelerator"] == "cu128"


def test_lerobot_run_auto_uses_native_settings_device_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_worker(env: object, args: list[str], **kwargs: object) -> Run:
        del env, args
        calls.append(kwargs)
        return Run(framework="lerobot", output_dir=tmp_path / "run", checkpoint=None)

    def fake_resolve(requested: str, *, allowed: tuple[str, ...]) -> str:
        assert requested == "cu128"
        assert "cu128" in allowed
        return "cu128"

    monkeypatch.setattr(lerobot_engine_module, "resolve_accelerator", fake_resolve)
    monkeypatch.setattr(lerobot_engine_module, "run_worker", fake_run_worker)

    LeRobotEngine.run(
        {"policy": {"type": "act", "device": "cuda"}},
        dataset=tmp_path / "snapshot",
        output_dir=tmp_path / "output",
        device="auto",
    )

    assert calls[0]["device"] == "cuda"
    assert calls[0]["accelerator"] == "cu128"


def test_native_settings_are_flattened_without_shell_interpretation() -> None:
    flags = settings_to_flags(
        {
            "policy": {"type": "act", "name": "pick up; echo unsafe"},
            "steps": 3,
            "resume": False,
            "dataset": {"episodes": [1, 2]},
        },
    )

    assert flags == [
        "--policy.type=act",
        "--policy.name=pick up; echo unsafe",
        "--steps=3",
        "--resume=false",
        "--dataset.episodes=[1,2]",
    ]

    with pytest.raises(ValueError, match="Invalid LeRobot setting name"):
        settings_to_flags({"../unsafe": True})

    deeply_nested: dict[str, object] = {"value": True}
    for _ in range(11):
        deeply_nested = {"nested": deeply_nested}
    with pytest.raises(ValueError, match="maximum nesting depth"):
        settings_to_flags(deeply_nested)


def test_lerobot_engine_import_does_not_import_training_dependencies() -> None:
    code = (
        "import sys; from physicalai_lerobot_policy_plugin.engine import LeRobotEngine; "
        "assert LeRobotEngine.framework == 'lerobot'; "
        "heavy = ('torch', 'lerobot', 'lightning'); "
        "assert not [name for name in sys.modules if any(name == item or name.startswith(item + '.') for item in heavy)]"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
