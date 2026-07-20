# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Launch command for the Physical AI trainer service.

This command loads the trainer ``.env`` file and starts the trainer service
in-process. Dependencies (including the hardware-specific torch build) must be
installed beforehand with ``uv sync --extra <cpu|cuda|xpu>``, so this command is
meant to be invoked with ``uv run physicalai-trainer``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import click

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Minimum length for a value to possibly be wrapped in a matching pair of quotes.
_MIN_QUOTED_LEN = 2


def _project_dir() -> Path:
    """Return the trainer project directory (contains pyproject.toml and .venv)."""
    # src/trainer/cli.py -> src/trainer -> src -> trainer
    return Path(__file__).resolve().parents[2]


def load_env_file(env_file: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``env_file`` without overriding real env vars.

    Variables already present in the environment win (matching Pydantic settings
    precedence), blank lines and ``#`` comments are ignored, a single layer of
    surrounding quotes is stripped, and a warning is emitted if the file is
    readable by group/other.
    """
    if not env_file.is_file():
        return

    try:
        mode = env_file.stat().st_mode & 0o777
    except OSError:
        mode = 0
    if mode & 0o077:
        click.echo(
            f"Warning: {env_file} is readable by group/other (mode {mode:03o}); Consider: chmod 600 {env_file}",
            err=True,
        )

    click.echo(f"Loading environment from {env_file}")
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export\s+", "", line)
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        if len(val) >= _MIN_QUOTED_LEN and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        # Only set when unset/empty so the caller's environment takes precedence.
        if not os.environ.get(key):
            os.environ[key] = val


@click.command()
@click.option("--host", default=None, help="Host to bind (defaults to settings).")
@click.option("--port", type=int, default=None, help="Port to bind (defaults to settings).")
def trainer(host: str | None, port: int | None) -> None:
    """Start the remote trainer service (run this on the GPU box).

    Install dependencies first with ``uv sync --extra <cpu|cuda|xpu>``.
    """
    project_dir = _project_dir()
    load_env_file(project_dir / ".env")

    os.environ["PYTHONUNBUFFERED"] = "1"

    click.echo("Starting remote trainer service...")
    import uvicorn

    from trainer.main import app
    from trainer.settings import get_settings

    # Refresh cached settings so values from the loaded .env take effect.
    get_settings.cache_clear()
    settings = get_settings()

    uvicorn.run(
        app,
        host=host if host is not None else settings.host,
        port=port if port is not None else settings.port,
    )


if __name__ == "__main__":
    trainer()
