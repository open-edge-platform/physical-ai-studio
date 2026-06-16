"""Imperative data and storage migrations.

These migrations move files on disk and rewrite stored paths in the database.
They are distinct from Alembic schema migrations (``src/alembic``), which change
the database structure. Run them from the CLI (``physicalai migrate`` and
``physicalai fix-dataset-paths``).
"""

from .dataset_path_migration import migrate_dataset_paths
from .storage_migration import StorageMigrationError, migrate_default_storage_dir

__all__ = [
    "StorageMigrationError",
    "migrate_dataset_paths",
    "migrate_default_storage_dir",
]
