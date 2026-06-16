"""Migrate existing datasets to unique, id-based storage folders.

Datasets used to derive their on-disk folder from the dataset name. Two datasets
with the same (sanitized) name - typically one per project - resolved to the same
folder, so episodes recorded in one project appeared in the other. This migration
gives every dataset its own ``<datasets_dir>/<dataset_id>`` folder.

A dataset whose path is already ``<datasets_dir>/<id>`` is left untouched. When
several datasets share a folder, the data is copied so each dataset keeps an
independent copy; otherwise the folder is moved.
"""

import shutil
from collections import defaultdict
from pathlib import Path

from loguru import logger
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from db.schema import DatasetDB
from settings import Settings


def migrate_dataset_paths(settings: Settings, *, dry_run: bool = False) -> int:
    """Move datasets to id-based folders so no two datasets share a directory.

    Args:
        settings: Application settings providing ``datasets_dir`` and the database URL.
        dry_run: When True, log the planned changes without touching disk or database.

    Returns:
        Number of datasets whose path was migrated.
    """
    database_path = settings.data_dir / settings.database_file
    if not database_path.exists():
        logger.info("No database found at {}; nothing to migrate.", database_path)
        return 0

    datasets_dir = settings.datasets_dir
    engine = create_engine(
        settings.database_url_sync,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    session_factory = sessionmaker(bind=engine)
    migrated = 0

    try:
        with session_factory() as session:
            rows = [(row_id, path) for row_id, path in session.query(DatasetDB.id, DatasetDB.path).all()]

            # Count how many datasets reference each resolved source folder so we know
            # whether a folder can be moved (single owner) or must be copied (shared).
            owners_per_path: dict[str, int] = defaultdict(int)
            for _row_id, path_value in rows:
                owners_per_path[_resolve(path_value)] += 1

            for row_id, path_value in rows:
                target = datasets_dir / row_id
                source = Path(path_value)

                if _resolve(path_value) == _resolve(str(target)):
                    continue  # Already id-based.

                shared = owners_per_path[_resolve(path_value)] > 1
                action = "copy" if shared else "move"
                logger.info(
                    "Migrating dataset {}: {} '{}' -> '{}'",
                    row_id,
                    action,
                    source,
                    target,
                )

                if dry_run:
                    migrated += 1
                    continue

                _relocate(source, target, copy=shared)
                session.execute(update(DatasetDB).where(DatasetDB.id == row_id).values(path=str(target)))
                migrated += 1

            if dry_run:
                session.rollback()
            else:
                session.commit()
    except Exception:
        logger.exception("Dataset path migration failed; no database changes were committed.")
        raise
    finally:
        engine.dispose()

    logger.info("Dataset path migration complete: {} dataset(s) {}.", migrated, "to migrate" if dry_run else "migrated")
    return migrated


def _resolve(path_value: str) -> str:
    try:
        return str(Path(path_value).expanduser().resolve())
    except OSError:
        return str(Path(path_value).expanduser().absolute())


def _relocate(source: Path, target: Path, *, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Target dataset folder already exists: {target}")

    if not source.exists():
        # Dataset row points at a missing folder. Create an empty target so the path
        # is valid; recording will populate it.
        logger.warning("Source dataset folder missing: {}. Creating empty target {}.", source, target)
        target.mkdir(parents=True)
        return

    if copy:
        shutil.copytree(source, target)
    else:
        shutil.move(str(source), str(target))
