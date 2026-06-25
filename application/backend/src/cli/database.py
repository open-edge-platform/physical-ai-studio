"""Database management CLI commands."""

import sys

import click


@click.group(name="db")
def database() -> None:
    """Database management commands."""


@database.command("init")
def init_db() -> None:
    """Initialize database with migrations"""
    from db.migration import MigrationManager
    from settings import get_settings

    click.echo("Initializing database...")

    migration_manager = MigrationManager(get_settings())
    if migration_manager.initialize_database():
        click.echo("✓ Database initialized successfully!")
        sys.exit(0)

    click.echo("✗ Database initialization failed!")
    sys.exit(1)


@database.command("clean")
def clean_db() -> None:
    """Remove all data from the database (clean but don't drop tables)."""
    from db.engine import get_sync_db_session
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


@database.command("check")
def check_db() -> None:
    """Check database status"""
    from db.migration import MigrationManager
    from settings import get_settings

    click.echo("Checking database status...")

    migration_manager = MigrationManager(get_settings())

    if not migration_manager.check_connection():
        click.echo("✗ Cannot connect to database")
        sys.exit(1)

    click.echo("✓ Database connection OK")

    needs_migration, status = migration_manager.check_migration_status()
    click.echo(f"Migration status: {status}")

    if needs_migration:
        click.echo("⚠ Database needs migration")
        sys.exit(2)

    click.echo("✓ Database is up to date")
    sys.exit(0)


@database.command("migrate")
def migrate() -> None:
    """Run database migrations."""
    from db.migration import MigrationManager
    from settings import get_settings
    from storage_migration import StorageMigrationError, migrate_default_storage_dir

    click.echo("Running database migrations...")
    settings = get_settings()

    try:
        migrate_default_storage_dir(settings)
    except StorageMigrationError as e:
        click.echo(f"✗ Storage migration failed: {e}", err=True)
        sys.exit(1)

    migration_manager = MigrationManager(settings)
    if not migration_manager.run_migrations():
        click.echo("✗ Migration failed!", err=True)
        sys.exit(1)
    click.echo("✓ Migrations completed successfully!")
