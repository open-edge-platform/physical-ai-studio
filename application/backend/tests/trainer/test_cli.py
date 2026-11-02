# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer launch CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import uvicorn
from click.testing import CliRunner

from trainer import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_load_env_file_sets_and_respects_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nexport SOME_SECRET='secret'\nTRAINER_STORAGE_DIR=\"/data\"\nALREADY_SET=fromfile\n"
        "not a valid line\n",
    )
    monkeypatch.delenv("SOME_SECRET", raising=False)
    monkeypatch.delenv("TRAINER_STORAGE_DIR", raising=False)
    monkeypatch.setenv("ALREADY_SET", "fromenv")

    cli.load_env_file(env_file)

    assert cli.os.environ["SOME_SECRET"] == "secret"
    assert cli.os.environ["TRAINER_STORAGE_DIR"] == "/data"
    # Existing environment value wins over the file.
    assert cli.os.environ["ALREADY_SET"] == "fromenv"


def test_load_env_file_missing_is_noop(tmp_path: Path) -> None:
    cli.load_env_file(tmp_path / "does-not-exist.env")


def test_trainer_command_reads_its_trainer_prefixed_settings_not_the_studios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The trainer must not inherit the studio backend's HOST/PORT/STORAGE_DIR.

    Both settings classes now read the same `.env`, so the trainer's own knobs
    use TRAINER_-prefixed names; reading the studio's bare PORT here would make
    the trainer bind the studio's port instead of its own.
    """
    (tmp_path / ".env").write_text("PORT=7860\nTRAINER_PORT=8001\n")
    monkeypatch.setattr(cli, "_project_dir", lambda: tmp_path)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("TRAINER_PORT", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(cli.trainer)

    assert result.exit_code == 0, result.output
    assert cli.os.environ["TRAINER_PORT"] == "8001"


def test_trainer_command_launches_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_project_dir", lambda: tmp_path)
    launched: dict[str, object] = {}

    def _fake_run(app: object, *, host: str, port: int, log_config: object = None) -> None:
        launched["host"] = host
        launched["port"] = port
        launched["log_config"] = log_config

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    result = CliRunner().invoke(cli.trainer, ["--host", "127.0.0.1", "--port", "9100"])

    assert result.exit_code == 0, result.output
    assert launched == {"host": "127.0.0.1", "port": 9100, "log_config": None}
