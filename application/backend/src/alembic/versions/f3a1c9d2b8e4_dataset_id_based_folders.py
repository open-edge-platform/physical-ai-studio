"""Move datasets to unique id-based storage folders

Datasets used to derive their on-disk folder from the dataset name. Two datasets
with the same (sanitized) name - typically one per project - resolved to the same
folder, so episodes recorded in one project appeared in the other. This migration
gives every dataset its own ``<datasets_dir>/<dataset_id>`` folder.

A dataset whose path is already ``<datasets_dir>/<id>`` is left untouched. When
several datasets share a folder, the data is copied so each dataset keeps an
independent copy; otherwise the folder is moved.

Revision ID: f3a1c9d2b8e4
Revises: e1f2a3b4c5d6
Create Date: 2026-06-19 00:00:00.000000

"""

import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import sqlalchemy as sa
from loguru import logger
from sqlalchemy import Connection

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a1c9d2b8e4"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Relocate every dataset into its own ``<datasets_dir>/<dataset_id>`` folder.

    Filesystem moves cannot be rolled back. Alembic runs this inside a transaction,
    so a failure mid-migration rolls back the path column updates while leaving
    relocated folders in place. Re-running the migration is safe: already id-based
    paths are skipped.
    """
    # Import here, not at module top: Alembic imports version files to read their
    # revision graph before env.py puts ``src`` on sys.path.
    from settings import Settings

    datasets_dir = Settings().datasets_dir.expanduser().resolve()
    relocate_dataset_paths(op.get_bind(), datasets_dir)


def downgrade() -> None:
    """Irreversible: the original name-based folders are not recoverable.

    The pre-migration layout shared folders between datasets, so reversing the
    relocation would reintroduce the cross-project episode leak this migration fixes.
    """


def relocate_dataset_paths(connection: Connection, datasets_dir: Path, *, dry_run: bool = False) -> int:
    """Move datasets to id-based folders so no two datasets share a directory.

    Args:
        connection: Database connection bound to the datasets table.
        datasets_dir: Resolved base directory that must contain every dataset folder.
        dry_run: When True, log the planned changes without touching disk or database.

    Returns:
        Number of datasets whose path was migrated.
    """
    rows = [(str(row_id), path) for row_id, path in connection.execute(sa.text("SELECT id, path FROM datasets"))]

    # Count how many datasets reference each resolved source folder so we know
    # whether a folder can be moved (single owner) or must be copied (shared).
    owners_per_path: dict[str, int] = defaultdict(int)
    for _row_id, path_value in rows:
        owners_per_path[_resolve(path_value)] += 1

    migrated = 0
    for row_id, path_value in rows:
        target = datasets_dir / row_id
        # Resolve to the canonical path so symlinks cannot redirect the relocate
        # outside datasets_dir.
        source = Path(_resolve(path_value))

        if _resolve(path_value) == _resolve(str(target)):
            continue  # Already id-based.

        # Refuse to touch anything outside datasets_dir. The path column was
        # historically client-supplied, so a malicious or corrupt row could
        # otherwise move/copy arbitrary filesystem locations.
        if not source.is_relative_to(datasets_dir):
            raise ValueError(
                f"Refusing to migrate dataset {row_id}: source path {source} escapes datasets_dir {datasets_dir}"
            )

        shared = owners_per_path[_resolve(path_value)] > 1
        action = "copy" if shared else "move"
        logger.info("Migrating dataset {}: {} '{}' -> '{}'", row_id, action, source, target)

        if dry_run:
            migrated += 1
            continue

        _relocate(source, target, copy=shared)
        connection.execute(
            sa.text("UPDATE datasets SET path = :path WHERE id = :id"),
            {"path": str(target), "id": row_id},
        )
        migrated += 1

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
