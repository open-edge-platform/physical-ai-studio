
"""Command line interface for interacting with the Geti Action application."""

import logging
import sys

import click

from db import MigrationManager, get_db_session
from db.schema import ProjectDB, ProjectConfigDB
from settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
migration_manager = MigrationManager(settings)


@click.group()
def cli() -> None:
    """Geti Action CLI"""


@cli.command()
def init_db() -> None:
    """Initialize database with migrations"""
    click.echo("Initializing database...")

    if migration_manager.initialize_database():
        click.echo("✓ Database initialized successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Database initialization failed!")
        sys.exit(1)


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_db_session() as db:
        db.query(ProjectDB).delete()
        db.query(ProjectConfigDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


@cli.command()
def migrate() -> None:
    """Run database migrations"""
    click.echo("Running database migrations...")

    if migration_manager.run_migrations():
        click.echo("✓ Migrations completed successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Migration failed!")
        sys.exit(1)


if __name__ == "__main__":
    cli()
