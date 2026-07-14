"""Command line interface for interacting with the Physical AI Studio application."""

import subprocess
import sys
from pathlib import Path

import click

from cli.database import database
from cli.models import models
from cli.run import remote
from cli.serve import serve
from robots.catalog.assets import get_builtin_robot_assets_root
from robots.catalog.sync_robot_assets import sync_robot_assets as sync_builtin_robot_assets


@click.group()
def cli() -> None:
    """Physical AI Studio CLI."""


@cli.command()
@click.option("--target-path", default="docs/openapi.json")
def gen_api(target_path: str) -> None:
    """Generate OpenAPI specification JSON file."""
    # Importing create_openapi imports threading which is slow. Importing here to not slow down other cli commands.
    from create_openapi import create_openapi

    try:
        create_openapi(target_path=target_path)
        click.echo("✓ OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


@cli.command()
@click.option(
    "--target-dir",
    default=str(get_builtin_robot_assets_root()),
    show_default=True,
)
def sync_robot_assets(target_dir: str) -> None:
    """Sync SO101 and WidowX assets into backend static storage."""
    try:
        sync_builtin_robot_assets(Path(target_dir))
        click.echo("✓ Robot assets synced successfully")
    except (subprocess.CalledProcessError, OSError) as error:
        click.echo(f"✗ Failed to sync robot assets: {error}")
        sys.exit(1)


cli.add_command(database)
cli.add_command(models)
cli.add_command(serve)
cli.add_command(remote)
