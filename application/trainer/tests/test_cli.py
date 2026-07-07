# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer launch CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest
import uvicorn
from click.testing import CliRunner

from trainer import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEVICE", raising=False)
    assert cli._resolve_device(None) == "cpu"


def test_resolve_device_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda")
    assert cli._resolve_device(None) == "cuda"


def test_resolve_device_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda")
    assert cli._resolve_device("xpu") == "xpu"


def test_resolve_device_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEVICE", raising=False)
    with pytest.raises(click.ClickException):
        cli._resolve_device("gpu")


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        (True, None, True),
        (False, "true", False),
        (None, "false", False),
        (None, "FALSE", False),
        (None, None, True),
        (None, "true", True),
    ],
)
def test_should_sync(monkeypatch: pytest.MonkeyPatch, flag: bool | None, env: str | None, expected: bool) -> None:
    if env is None:
        monkeypatch.delenv("SYNC", raising=False)
    else:
        monkeypatch.setenv("SYNC", env)
    assert cli._should_sync(flag) is expected


def test_load_env_file_sets_and_respects_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nexport HF_TOKEN='secret'\nSTORAGE_DIR=\"/data\"\nALREADY_SET=fromfile\nnot a valid line\n",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    monkeypatch.setenv("ALREADY_SET", "fromenv")

    cli.load_env_file(env_file)

    assert cli.os.environ["HF_TOKEN"] == "secret"
    assert cli.os.environ["STORAGE_DIR"] == "/data"
    # Existing environment value wins over the file.
    assert cli.os.environ["ALREADY_SET"] == "fromenv"


def test_load_env_file_missing_is_noop(tmp_path: Path) -> None:
    cli.load_env_file(tmp_path / "does-not-exist.env")


def test_maybe_sync_skips_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False

    def _fail(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli.subprocess, "run", _fail)
    cli.maybe_sync(tmp_path, "cpu", sync=False)
    assert called is False


def test_maybe_sync_runs_uv_sync(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_args: list[str] = []
    captured_kwargs: dict[str, object] = {}

    def _capture(args: list[str], **kwargs: object) -> None:
        captured_args.extend(args)
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(cli.subprocess, "run", _capture)
    cli.maybe_sync(tmp_path, "cuda", sync=True)
    assert captured_args == ["uv", "sync", "--extra", "cuda"]
    assert captured_kwargs["cwd"] == tmp_path


def test_trainer_command_launches_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_project_dir", lambda: tmp_path)
    monkeypatch.setenv("HF_TOKEN", "token")
    launched: dict[str, object] = {}

    def _fake_run(app: object, *, host: str, port: int) -> None:
        launched["host"] = host
        launched["port"] = port

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    result = CliRunner().invoke(cli.trainer, ["--no-sync", "--host", "127.0.0.1", "--port", "9100"])

    assert result.exit_code == 0, result.output
    assert launched == {"host": "127.0.0.1", "port": 9100}


def test_trainer_command_warns_without_hf_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_project_dir", lambda: tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: None)

    result = CliRunner().invoke(cli.trainer, ["--no-sync"])

    assert result.exit_code == 0, result.output
    assert "HF_TOKEN is not set" in result.output
