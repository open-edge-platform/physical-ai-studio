"""Command line interface for interacting with the Physical AI Studio application."""

import sys

import click

from cli.database import database
from cli.datasets import datasets
from cli.hf import hf
from cli.models import models
from cli.serve import serve


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "\b\nGuide: https://github.com/open-edge-platform/physical-ai-studio/"
        "blob/main/application/backend/docs/studio_cli.md"
    ),
)
def cli() -> None:
    """Manage the local Physical AI Studio application and its stored assets.

    Use the same STORAGE_DIR, SETTINGS_FILE, and database configuration as Studio.
    These commands do not connect to a remote Studio HTTP server. In Docker, run
    them inside the backend container.

    Start with COMMAND --help. Use hf for Hub transfers, models/datasets for local
    folder imports, and db for database maintenance. Policy training and inference
    format conversion belong to the separate physicalai CLI.
    """


@cli.command()
@click.option(
    "--target-path",
    default="docs/openapi.json",
    show_default=True,
    help="Output JSON path; an existing file is overwritten.",
)
def gen_api(target_path: str) -> None:
    """Generate OpenAPI JSON for development tooling.

    Imports the application and overwrites the target JSON file if it exists.
    """
    # Importing create_openapi imports threading which is slow. Importing here to not slow down other cli commands.
    from create_openapi import create_openapi

    try:
        create_openapi(target_path=target_path)
        click.echo("✓ OpenAPI specification generated successfully!")
    except Exception as e:
        click.echo(f"✗ Failed to generate OpenAPI specification: {e}")
        sys.exit(1)
    click.echo("Waiting for threading to finish...")


cli.add_command(database)
cli.add_command(datasets)
cli.add_command(hf)
cli.add_command(models)
cli.add_command(serve)
