import sqlite3
from pathlib import Path

import pytest

from alembic import command
from db.migration import MigrationManager
from settings import Settings


def test_aws_singleton_constraint_and_downgrade(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    configuration = MigrationManager(settings).get_alembic_config()
    command.upgrade(configuration, "7e2a9c4d5b61")
    database = settings.data_dir / settings.database_file
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, connection) VALUES (?, ?, ?, ?)",
            ("aws", "Custom name", "aws_batch", "{}"),
        )
    command.upgrade(configuration, "91d5e3a7c2b0")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name FROM remote_trainers WHERE id = 'aws'").fetchone() == ("AWS Provider",)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO remote_trainers (id, name, connection_mode, connection) VALUES (?, ?, ?, ?)",
                ("another", "Different name", "aws_batch", "{}"),
            )
        for identifier in ("direct-1", "direct-2", "ssh-1", "ssh-2"):
            connection.execute(
                "INSERT INTO remote_trainers (id, name, connection_mode, connection) VALUES (?, ?, ?, ?)",
                (identifier, identifier, identifier.split("-")[0], "{}"),
            )
    command.downgrade(configuration, "7e2a9c4d5b61")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, connection) VALUES (?, ?, ?, ?)",
            ("another", "Different name", "aws_batch", "{}"),
        )


def test_migration_rejects_existing_duplicates_without_deleting_them(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    configuration = MigrationManager(settings).get_alembic_config()
    command.upgrade(configuration, "7e2a9c4d5b61")
    database = settings.data_dir / settings.database_file
    with sqlite3.connect(database) as connection:
        for identifier in ("first", "second"):
            connection.execute(
                "INSERT INTO remote_trainers (id, name, connection_mode, connection) VALUES (?, ?, ?, ?)",
                (identifier, identifier, "aws_batch", "{}"),
            )
    with pytest.raises(RuntimeError, match="Multiple AWS providers"):
        command.upgrade(configuration, "91d5e3a7c2b0")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM remote_trainers").fetchone() == (2,)
