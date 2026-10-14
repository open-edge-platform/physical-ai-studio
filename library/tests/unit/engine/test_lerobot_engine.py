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


def test_lerobot_cli_inherits_stdio_and_returns_native_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(lerobot_engine_module, "interpreter_for", lambda *args, **kwargs: Path("/env/bin/python"))
    def fake_call(command: list[str], *, shell: bool) -> int:
        calls.append((command, shell))
        return 2

    monkeypatch.setattr(lerobot_engine_module.subprocess, "call", fake_call)

    result = LeRobotEngine.cli(["--help", "--policy.type=act"], accelerator="xpu")

    assert result == 2
    assert calls == [(["/env/bin/lerobot-train", "--help", "--policy.type=act"], False)]


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

    result = LeRobotEngine(accelerator="xpu").train(_Config(), hooks=hooks)  # type: ignore[arg-type]

    assert result is expected_run
    assert events[0][0] == "saved"
    assert events[1][0] == "run"
    assert events[1][3:] == ("xpu", hooks, "xpu", None, Version("0.6.1"))
    assert not config_file.exists()


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
