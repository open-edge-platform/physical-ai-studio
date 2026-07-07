# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Launch command for the Physical AI trainer service.

This command loads the trainer ``.env`` file, syncs dependencies for the
requested hardware (``uv sync``), and then starts the trainer service in-process.

Because it runs ``uv sync`` for itself, it is meant to be invoked through
``uv run`` (e.g. ``uv run --no-sync physicalai-trainer``) so that the base
dependencies are already available.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import click

_VALID_DEVICES = ("cpu", "cuda", "xpu")
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
    surrounding quotes is stripped, and a warning is emitted if the file (which
    may hold ``HF_TOKEN``) is readable by group/other.
    """
    if not env_file.is_file():
        return

    try:
        mode = env_file.stat().st_mode & 0o777
    except OSError:
        mode = 0
    if mode & 0o077:
        click.echo(
            f"Warning: {env_file} is readable by group/other (mode {mode:03o}); "
            f"it may contain HF_TOKEN. Consider: chmod 600 {env_file}",
            err=True,
        )

    click.echo(f"Loading environment from {env_file}")
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
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


def _resolve_device(device: str | None) -> str:
    """Resolve the hardware extra to sync from the flag or the DEVICE env var."""
    resolved = (device or os.environ.get("DEVICE") or "cpu").lower()
    if resolved not in _VALID_DEVICES:
        msg = f"DEVICE must be one of {', '.join(_VALID_DEVICES)} (got '{resolved}')."
        raise click.ClickException(msg)
    return resolved


def _should_sync(sync: bool | None) -> bool:
    """Resolve whether to run ``uv sync`` from the flag or the SYNC env var."""
    if sync is not None:
        return sync
    return os.environ.get("SYNC", "true").lower() != "false"


def maybe_sync(cwd: Path, device: str, *, sync: bool | None) -> None:
    """Run ``uv sync`` for ``device`` in ``cwd`` unless disabled."""
    if not _should_sync(sync):
        click.echo("Skipping dependency sync (SYNC=false).")
        return

    args = ["uv", "sync", "--extra", device]
    click.echo(f"Syncing dependencies: {shlex.join(args)}")
    subprocess.run(args, cwd=cwd, check=True)  # noqa: S603 - fixed argv, no shell.


@click.command()
@click.option(
    "--device",
    type=click.Choice(_VALID_DEVICES),
    default=None,
    help="Hardware extra to sync (defaults to $DEVICE or cpu).",
)
@click.option(
    "--sync/--no-sync",
    "sync",
    default=None,
    help="Run `uv sync` before launching (defaults to $SYNC or true).",
)
@click.option("--host", default=None, help="Host to bind (defaults to settings).")
@click.option("--port", type=int, default=None, help="Port to bind (defaults to settings).")
def trainer(device: str | None, sync: bool | None, host: str | None, port: int | None) -> None:
    """Start the remote trainer service (run this on the GPU box)."""
    project_dir = _project_dir()
    load_env_file(project_dir / ".env")
    resolved_device = _resolve_device(device)

    if not os.environ.get("HF_TOKEN"):
        click.echo("Warning: HF_TOKEN is not set; the trainer cannot pull dataset snapshots.", err=True)

    os.environ["PYTHONUNBUFFERED"] = "1"
    maybe_sync(project_dir, resolved_device, sync=sync)

    click.echo("Starting remote trainer service...")
    # Import after syncing so freshly-installed hardware-specific deps are used.
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
