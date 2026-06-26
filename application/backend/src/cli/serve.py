"""Serve backend and frontend CLI commands."""

import os
from pathlib import Path

import click

from cli.database import _run_migrations
from settings import get_settings

settings = get_settings()


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _configure_packaged_runtime() -> None:
    package_root = _package_root()
    bundled_webui = package_root / "webui"

    if bundled_webui.joinpath("index.html").exists():
        os.environ.setdefault("STATIC_FILES_DIR", str(bundled_webui))

    os.environ.setdefault("ALEMBIC_CONFIG_PATH", str(package_root / "alembic.ini"))
    os.environ.setdefault("ALEMBIC_SCRIPT_LOCATION", str(package_root / "alembic"))


@click.command()
@click.option("--host", default=settings.host, show_default=True)
@click.option("--port", default=settings.port, show_default=True, type=int)
def serve(host: str, port: int) -> None:
    """Start the Physical AI Studio web application."""
    _configure_packaged_runtime()
    _run_migrations()

    import uvicorn

    from utils.multiprocessing import ensure_spawn_start_method

    ensure_spawn_start_method()
    uvicorn.run("main:app", host=host, port=port)
