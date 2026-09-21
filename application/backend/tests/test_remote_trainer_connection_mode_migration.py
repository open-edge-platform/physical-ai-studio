"""Functional test for the remote trainer connection-mode migration."""

import sqlite3
from pathlib import Path

from alembic import command
from db.migration import MigrationManager
from settings import Settings

_PRE_MIGRATION_REVISION = "c9d1e2f3a4b5"
_MIGRATION_REVISION = "4b8d2f6a1c30"


def test_migration_adds_explicit_ssh_fields_and_backfills_alias_mode(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    alembic_cfg = MigrationManager(settings).get_alembic_config()
    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    db_path = settings.data_dir / settings.database_file
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO remote_trainers (id, name, url, ssh_host_alias, ssh_remote_port, ssh_local_port) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("trainer-id", "trainer", "http://127.0.0.1:8001", "gpu-box", 8001, 8001),
        )
        connection.execute(
            "INSERT INTO remote_trainers (id, name, url) VALUES (?, ?, ?)",
            ("direct-trainer-id", "direct trainer", "https://trainer.example.test"),
        )
        connection.commit()

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(remote_trainers)")}
        modes = dict(connection.execute("SELECT id, connection_mode FROM remote_trainers").fetchall())

    assert {"connection_mode", "ssh_hostname", "ssh_port", "ssh_username", "ssh_identity_file"} <= columns
    assert modes == {"trainer-id": "ssh", "direct-trainer-id": "direct"}
