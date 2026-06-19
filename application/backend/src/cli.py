"""Command line interface for interacting with the Physical AI Studio application."""

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import click

from db.engine import get_sync_db_session
from db.migration import MigrationManager
from db.schema import (
    CalibrationValuesDB,
    DatasetDB,
    JobDB,
    ProjectCameraDB,
    ProjectDB,
    ProjectEnvironmentDB,
    ProjectRobotDB,
    SnapshotDB,
)
from settings import get_settings
from storage_migration import StorageMigrationError, migrate_default_storage_dir

settings = get_settings()


@click.group()
def cli() -> None:
    """Physical AI Studio CLI"""


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
def init_db() -> None:
    """Initialize database with migrations"""
    click.echo("Initializing database...")

    migration_manager = MigrationManager(settings)
    if migration_manager.initialize_database():
        click.echo("✓ Database initialized successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Database initialization failed!")
        sys.exit(1)


@cli.command()
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    with get_sync_db_session() as db:
        db.query(ProjectDB).delete()
        db.query(ProjectRobotDB).delete()
        db.query(CalibrationValuesDB).delete()
        db.query(ProjectCameraDB).delete()
        db.query(ProjectEnvironmentDB).delete()
        db.query(DatasetDB).delete()
        db.query(SnapshotDB).delete()
        db.query(JobDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


@cli.command()
def check_db() -> None:
    """Check database status"""
    click.echo("Checking database status...")

    migration_manager = MigrationManager(settings)

    # Check connection
    if not migration_manager.check_connection():
        click.echo("✗ Cannot connect to database")
        sys.exit(1)

    click.echo("✓ Database connection OK")

    # Check migration status
    needs_migration, status = migration_manager.check_migration_status()
    click.echo(f"Migration status: {status}")

    if needs_migration:
        click.echo("⚠ Database needs migration")
        sys.exit(2)
    else:
        click.echo("✓ Database is up to date")
        sys.exit(0)


@cli.command()
def migrate() -> None:
    """Run database migrations"""
    click.echo("Running database migrations...")

    try:
        migrate_default_storage_dir(settings)
    except StorageMigrationError as e:
        click.echo(f"✗ Storage migration failed: {e}", err=True)
        sys.exit(1)

    migration_manager = MigrationManager(settings)
    if migration_manager.run_migrations():
        click.echo("✓ Migrations completed successfully!")
        sys.exit(0)
    else:
        click.echo("✗ Migration failed!")
        sys.exit(1)


# =============================================================================
# Models subcommand group
# =============================================================================


@cli.group()
def models() -> None:
    """Model management commands."""


@models.command("import-dir")
@click.option("--source-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--project-id", required=True, type=click.UUID)
@click.option("--dataset-id", required=True, type=click.UUID)
@click.option("--model-name", required=True, type=str)
@click.option("--move/--copy", default=False, show_default=True)
@click.option("--base-model-id", type=click.UUID, default=None)
@click.option("--version", type=int, default=1, show_default=True)
def import_dir(
    source_dir: Path,
    project_id: UUID,
    dataset_id: UUID,
    model_name: str,
    move: bool,
    base_model_id: UUID | None,
    version: int,
) -> None:
    """Import a model from an existing folder (copy or move) trained by Physical AI Studio.

    The dataset_id should reference a dataset that uses the same environment as the
    original training. The inference UI uses this dataset to determine which environment
    to load when running the model.
    """
    from services.model_import_service import ModelImportService

    click.echo(f"Importing model from folder: {source_dir}")
    click.echo(f"Mode: {'move' if move else 'copy'}")

    async def _run_import() -> None:
        service = ModelImportService()
        model = await service.import_model_directory(
            source_dir=source_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            move=move,
            base_model_id=base_model_id,
            version=version,
        )
        click.echo("Model imported successfully!")
        click.echo(f"Model ID: {model.id}")
        click.echo(f"Model path: {model.path}")

    try:
        asyncio.run(_run_import())
    except Exception as e:
        click.echo(f"Model import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
