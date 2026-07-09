"""Launch commands for Physical AI Studio components.

The ``remote`` command loads the matching ``.env`` file, syncs dependencies for the
requested hardware (``uv sync``), and then starts the backend with training offloaded
to a remote trainer service. In-process training is the default and is served by
``physicalai-studio serve``. The remote trainer service has its own launcher
(``physicalai-trainer``) in the trainer project.

Because these commands run ``uv sync`` for themselves, they are meant to be
invoked through ``uv run`` (e.g. ``uv run --no-sync physicalai-studio remote``)
so that the base dependencies are already available.
"""

import os
import re
import shlex
import subprocess
from pathlib import Path

import click

_VALID_DEVICES = ("cpu", "cuda", "xpu")
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXPORT_PREFIX_RE = re.compile(r"^export\s+")


def _backend_dir() -> Path:
    """Return the backend project directory (contains pyproject.toml and .venv)."""
    # src/cli/run.py -> src/cli -> src -> backend
    return Path(__file__).resolve().parents[2]


def load_env_file(env_file: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``env_file`` without overriding real env vars.

    Variables already present in the environment win (matching
    Pydantic settings precedence), blank lines and ``#`` comments are ignored, a
    single layer of surrounding quotes is stripped, and a warning is emitted if the
    file (which may hold ``HF_TOKEN``) is readable by group/other.
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
        line = _EXPORT_PREFIX_RE.sub("", line)
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]
        # Only set when unset/empty so the caller's environment takes precedence.
        if not os.environ.get(key):
            os.environ[key] = val


def _resolve_device(device: str | None) -> tuple[str, bool]:
    """Resolve the hardware extra and whether it was explicitly requested."""
    env_device = os.environ.get("DEVICE")
    explicit = device is not None or bool(env_device)
    resolved = (device or env_device or "cpu").lower()
    if resolved not in _VALID_DEVICES:
        raise click.ClickException(f"DEVICE must be one of {', '.join(_VALID_DEVICES)} (got '{resolved}').")
    return resolved, explicit


def _should_sync(sync: bool | None) -> bool:
    """Resolve whether to run ``uv sync`` from the flag or the SYNC env var."""
    if sync is not None:
        return sync
    return os.environ.get("SYNC", "true").lower() != "false"


def maybe_sync(cwd: Path, device: str, *extras: str, sync: bool | None) -> None:
    """Run ``uv sync`` for ``device`` (plus ``extras``) in ``cwd`` unless disabled."""
    if not _should_sync(sync):
        click.echo("Skipping dependency sync (SYNC=false).")
        return

    args = ["uv", "sync", "--extra", device, *(arg for extra in extras for arg in ("--extra", extra))]
    click.echo(f"Syncing dependencies: {shlex.join(args)}")
    try:
        subprocess.run(args, cwd=cwd, check=True)  # noqa: S603 - fixed argv, no shell.
    except (subprocess.CalledProcessError, OSError) as error:
        raise click.ClickException(f"Failed to sync dependencies ({shlex.join(args)}): {error}") from error


_device_option = click.option(
    "--device",
    type=click.Choice(_VALID_DEVICES),
    default=None,
    help="Hardware extra to sync (defaults to $DEVICE or cpu).",
)
_sync_option = click.option(
    "--sync/--no-sync",
    "sync",
    default=None,
    help="Run `uv sync` before launching (defaults to $SYNC or true).",
)
_host_option = click.option("--host", default=None, help="Host to bind (defaults to settings).")
_port_option = click.option("--port", type=int, default=None, help="Port to bind (defaults to settings).")


@click.command()
@_host_option
@_port_option
@_device_option
@_sync_option
def remote(host: str | None, port: int | None, device: str | None, sync: bool | None) -> None:
    """Start the backend with training offloaded to a remote trainer service."""
    load_env_file(_backend_dir() / ".env")
    resolved_device, device_explicit = _resolve_device(device)

    os.environ["TRAINING_MODE"] = "remote"
    os.environ["PYTHONUNBUFFERED"] = "1"

    if not os.environ.get("TRAINER_URL"):
        raise click.ClickException(
            "'remote' mode requires TRAINER_URL to point at a running trainer service.\n"
            "Example: TRAINER_URL=http://gpu-host:8001 physicalai-studio remote",
        )

    # Remote/recording nodes offload training, so cpu torch suffices; a GPU extra
    # only matters for local torch-backend GPU inference.
    if device_explicit and resolved_device != "cpu":
        click.echo(
            f"Note: DEVICE={resolved_device} on a remote node only affects local torch-backend "
            "GPU inference; training is offloaded, so cpu torch suffices for recording.",
        )
    maybe_sync(_backend_dir(), resolved_device, sync=sync)

    from settings import get_settings

    settings = get_settings()
    from cli.serve import start_server

    resolved_host = host if host is not None else settings.host
    resolved_port = port if port is not None else settings.port
    start_server(resolved_host, resolved_port)
