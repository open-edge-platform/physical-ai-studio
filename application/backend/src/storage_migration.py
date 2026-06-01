"""Filesystem migration for the persistent storage directory."""

import os
import shutil
import sys
from pathlib import Path

import click
from loguru import logger
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.pool import NullPool

from db.schema import DatasetDB, ModelDB, RobotCalibrationDB, SnapshotDB
from settings import Settings

OLD_DEFAULT_STORAGE_DIR = Path("~/.cache/physicalai").expanduser()
AUTO_MIGRATE_STORAGE_DIR_ENV = "AUTO_MIGRATE_STORAGE_DIR"
EXPECTED_STORAGE_SUBDIRS = {"models", "cache", "snapshots", "datasets", "robots", "logs"}


class StorageMigrationError(Exception):
    """Raised when storage migration cannot be completed safely."""


def migrate_default_storage_dir(settings: Settings, *, interactive: bool | None = None) -> None:
    """Move data from the old cache-backed default storage directory when safe."""
    old_storage_dir = OLD_DEFAULT_STORAGE_DIR
    new_storage_dir = settings.storage_dir.expanduser()

    if not old_storage_dir.exists():
        return

    if _same_path(old_storage_dir, new_storage_dir):
        return

    auto_migrate = _env_flag(AUTO_MIGRATE_STORAGE_DIR_ENV)
    storage_dir_is_overridden = "STORAGE_DIR" in os.environ
    if storage_dir_is_overridden and not auto_migrate:
        logger.info("Skipping storage migration because STORAGE_DIR is set.")
        return

    if not _is_effectively_empty_storage_dir(new_storage_dir):
        raise StorageMigrationError(_non_empty_destination_message(old_storage_dir, new_storage_dir))

    if interactive is None:
        interactive = sys.stdin.isatty()

    if not auto_migrate:
        if not interactive:
            raise StorageMigrationError(_non_interactive_message(old_storage_dir, new_storage_dir))

        click.echo("Physical AI Studio storage needs to move to a persistent application data directory.")
        click.echo(f"  from: {old_storage_dir}")
        click.echo(f"  to:   {new_storage_dir}")
        if not click.confirm("Move existing storage now?", default=False):
            raise StorageMigrationError(_declined_message(old_storage_dir, new_storage_dir))

    logger.info(f"Migrating storage directory from {old_storage_dir} to {new_storage_dir}")
    _remove_empty_storage_scaffold(new_storage_dir)
    new_storage_dir.parent.mkdir(parents=True, exist_ok=True)
    _move_storage_dir(old_storage_dir, new_storage_dir)
    _rewrite_database_paths(settings, old_storage_dir, new_storage_dir)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.expanduser().absolute() == right.expanduser().absolute()


def _is_effectively_empty_storage_dir(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        return False

    for child in path.iterdir():
        if child.name not in EXPECTED_STORAGE_SUBDIRS or not child.is_dir():
            return False
        if any(child.iterdir()):
            return False
    return True


def _remove_empty_storage_scaffold(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _move_storage_dir(source: Path, destination: Path) -> None:
    try:
        source.rename(destination)
    except OSError:
        shutil.move(str(source), str(destination))


def _rewrite_database_paths(settings: Settings, old_storage_dir: Path, new_storage_dir: Path) -> None:
    database_path = settings.data_dir / settings.database_file
    if not database_path.exists():
        return

    engine = create_engine(
        settings.database_url_sync,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        try:
            _rewrite_table_paths(session, DatasetDB, DatasetDB.path, old_storage_dir, new_storage_dir)
            _rewrite_table_paths(session, ModelDB, ModelDB.path, old_storage_dir, new_storage_dir)
            _rewrite_table_paths(session, SnapshotDB, SnapshotDB.path, old_storage_dir, new_storage_dir)
            _rewrite_table_paths(
                session, RobotCalibrationDB, RobotCalibrationDB.file_path, old_storage_dir, new_storage_dir
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
    engine.dispose()


def _rewrite_table_paths(
    session: Session,
    model: type[DatasetDB] | type[ModelDB] | type[SnapshotDB] | type[RobotCalibrationDB],
    column: InstrumentedAttribute[str],
    old_storage_dir: Path,
    new_storage_dir: Path,
) -> None:
    for row_id, path_value in session.query(model.id, column).all():
        rewritten_path = _rewrite_path_prefix(path_value, old_storage_dir, new_storage_dir)
        if rewritten_path != path_value:
            session.execute(update(model).where(model.id == row_id).values({column.key: rewritten_path}))


def _rewrite_path_prefix(path_value: str, old_storage_dir: Path, new_storage_dir: Path) -> str:
    try:
        relative_path = Path(path_value).relative_to(old_storage_dir)
    except ValueError:
        return path_value
    return str(new_storage_dir / relative_path)


def _non_empty_destination_message(old_storage_dir: Path, new_storage_dir: Path) -> str:
    return (
        "Cannot automatically migrate Physical AI Studio storage because the new storage directory already "
        "contains files.\n\n"
        f"Old storage: {old_storage_dir}\n"
        f"New storage: {new_storage_dir}\n\n"
        f"Move or merge the data manually, or set STORAGE_DIR={old_storage_dir} to keep using the old location."
    )


def _non_interactive_message(old_storage_dir: Path, new_storage_dir: Path) -> str:
    return (
        "Physical AI Studio storage needs to move, but startup is non-interactive.\n\n"
        f"Old storage: {old_storage_dir}\n"
        f"New storage: {new_storage_dir}\n\n"
        f"Move the directory manually, set STORAGE_DIR={old_storage_dir} to keep using the old location, or set "
        f"{AUTO_MIGRATE_STORAGE_DIR_ENV}=true to allow automatic migration."
    )


def _declined_message(old_storage_dir: Path, new_storage_dir: Path) -> str:
    return (
        "Storage migration was declined.\n\n"
        f"Old storage: {old_storage_dir}\n"
        f"New storage: {new_storage_dir}\n\n"
        f"Move the directory manually or set STORAGE_DIR={old_storage_dir} to keep using the old location."
    )
