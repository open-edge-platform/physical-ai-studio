"""Database management CLI commands."""

import sys

import click


@click.group(name="db")
def database() -> None:
    """Check or maintain the configured local Studio database.

    Confirm STORAGE_DIR and DATABASE_FILE before any maintenance. init/migrate can
    upgrade the schema; clean is destructive. Back up existing data first.
    """


def _run_migrations() -> None:
    from db.migration import MigrationManager
    from settings import get_settings
    from storage_migration import StorageMigrationError, migrate_default_storage_dir

    settings = get_settings()

    try:
        migrate_default_storage_dir(settings)
    except StorageMigrationError as e:
        click.echo(f"✗ Storage migration failed: {e}", err=True)
        sys.exit(1)

    migration_manager = MigrationManager(settings)
    if not migration_manager.initialize_database():
        click.echo("✗ Migration failed!", err=True)
        sys.exit(1)


@database.command("init")
def init_db() -> None:
    """Initialize or upgrade the configured database using schema migrations.

    Creates storage directories and applies pending migrations to existing data.
    Do not use this to work around an unexpectedly missing project; first check
    the storage/database configuration.
    """
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
    """Delete project data and jobs from the database (DESTRUCTIVE).

    Does not remove the schema or stored asset files. Runs without a confirmation
    prompt. Never use this to repair a configuration mismatch; back up first.
    """
    from db.engine import get_sync_db_session
    from db.schema import DatasetDB, JobDB, ProjectCameraDB, ProjectDB, ProjectEnvironmentDB, ProjectRobotDB, SnapshotDB

    with get_sync_db_session() as db:
        db.query(ProjectDB).delete()
        db.query(ProjectRobotDB).delete()
        db.query(ProjectCameraDB).delete()
        db.query(ProjectEnvironmentDB).delete()
        db.query(DatasetDB).delete()
        db.query(SnapshotDB).delete()
        db.query(JobDB).delete()
        db.commit()
    click.echo("✓ Database cleaned successfully!")


@database.command("check")
def check_db() -> None:
    """Check connectivity and migration status without applying migrations.

    May create storage directories and an empty SQLite file if they do not exist.
    Exit codes: 0 = current schema, 1 = connection/error, 2 = migrations needed
    or migration status could not be determined.
    """
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
    """Migrate legacy storage if needed and upgrade the database schema.

    Modifies the configured local installation. Back up data before upgrading.
    """
    click.echo("Running database migrations...")
    _run_migrations()
    click.echo("✓ Migrations completed successfully!")
