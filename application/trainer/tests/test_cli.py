# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer launch CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import uvicorn
from click.testing import CliRunner

from trainer import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_load_env_file_sets_and_respects_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nexport SOME_SECRET='secret'\nSTORAGE_DIR=\"/data\"\nALREADY_SET=fromfile\nnot a valid line\n",
    )
    monkeypatch.delenv("SOME_SECRET", raising=False)
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    monkeypatch.setenv("ALREADY_SET", "fromenv")

    cli.load_env_file(env_file)

    assert cli.os.environ["SOME_SECRET"] == "secret"
    assert cli.os.environ["STORAGE_DIR"] == "/data"
    # Existing environment value wins over the file.
    assert cli.os.environ["ALREADY_SET"] == "fromenv"


def test_load_env_file_missing_is_noop(tmp_path: Path) -> None:
    cli.load_env_file(tmp_path / "does-not-exist.env")


def test_trainer_command_launches_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_project_dir", lambda: tmp_path)
    launched: dict[str, object] = {}

    def _fake_run(app: object, *, host: str, port: int) -> None:
        launched["host"] = host
        launched["port"] = port

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    result = CliRunner().invoke(cli.trainer, ["--host", "127.0.0.1", "--port", "9100"])

    assert result.exit_code == 0, result.output
    assert launched == {"host": "127.0.0.1", "port": 9100}
