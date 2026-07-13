"""Serve backend and frontend CLI commands."""

import os
import subprocess
import sys
from pathlib import Path

import click

from cli.database import _run_migrations
from robots.catalog.assets import builtin_robot_assets_are_available
from robots.catalog.sync_robot_assets import sync_robot_assets
from settings import get_settings


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configure_packaged_runtime() -> None:
    package_root = _package_root()
    bundled_webui = package_root / "webui"

    if bundled_webui.joinpath("index.html").exists():
        os.environ.setdefault("STATIC_FILES_DIR", str(bundled_webui))

    os.environ.setdefault("ALEMBIC_CONFIG_PATH", str(package_root / "alembic.ini"))
    os.environ.setdefault("ALEMBIC_SCRIPT_LOCATION", str(package_root / "alembic"))

    # Refresh cached settings so downstream DB/migration modules observe packaged paths.
    get_settings.cache_clear()


@click.command()
@click.option("--host", default=lambda: get_settings().host, show_default=True)
@click.option("--port", default=lambda: get_settings().port, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Start the Physical AI Studio web application."""
    _configure_packaged_runtime()
    _sync_missing_robot_assets()
    _run_migrations()

    import uvicorn

    from utils.multiprocessing import ensure_spawn_start_method

    ensure_spawn_start_method()
    uvicorn.run("main:app", host=host, port=port)


def _sync_missing_robot_assets() -> None:
    if builtin_robot_assets_are_available():
        return

    click.echo("Robot assets are missing; syncing them now...")
    try:
        sync_robot_assets()
    except (subprocess.CalledProcessError, OSError) as error:
        click.echo(f"✗ Failed to sync robot assets: {error}")
        sys.exit(1)
