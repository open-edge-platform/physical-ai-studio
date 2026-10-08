"""Functional test for folding remote trainer connection columns into JSON."""

import json
import sqlite3
from pathlib import Path

from alembic import command
from db.migration import MigrationManager
from settings import Settings

_PRE_MIGRATION_REVISION = "7c1a9e2b4d6f"
_MIGRATION_REVISION = "7e2a9c4d5b61"


def _upgrade_to_pre_migration(tmp_path: Path):
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    alembic_cfg = MigrationManager(settings).get_alembic_config()
    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)
    return alembic_cfg, settings.data_dir / settings.database_file


def test_migration_folds_columns_into_connection_json(tmp_path: Path) -> None:
    alembic_cfg, db_path = _upgrade_to_pre_migration(tmp_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, url, ssh_host_alias, ssh_remote_port, "
            "ssh_local_port) VALUES ('alias', 'alias trainer', 'ssh', 'http://127.0.0.1:9001', 'gpu-box', 8001, 9001)"
        )
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, url, ssh_hostname, ssh_port, ssh_username, "
            "ssh_identity_file, ssh_remote_port, ssh_local_port) VALUES ('manual', 'manual trainer', 'ssh', "
            "'http://127.0.0.1:8001', 'gpu.example.test', 2222, 'trainer', '~/.ssh/t', 8001, 8001)"
        )
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, url) "
            "VALUES ('direct', 'direct trainer', 'direct', 'https://trainer.example.test')"
        )
        connection.commit()

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(remote_trainers)")}
        rows = {
            row[0]: (row[1], json.loads(row[2]))
            for row in connection.execute("SELECT id, connection_mode, connection FROM remote_trainers")
        }

    assert "connection" in columns
    assert not columns & {"url", "ssh_host_alias", "ssh_hostname", "ssh_port", "ssh_remote_port", "ssh_local_port"}
    assert rows["direct"] == ("direct", {"connection_mode": "direct", "url": "https://trainer.example.test"})
    assert rows["alias"] == (
        "ssh",
        {
            "connection_mode": "ssh",
            "ssh_host_alias": "gpu-box",
            "ssh_connection": None,
            "ssh_remote_port": 8001,
            "ssh_local_port": 9001,
        },
    )
    assert rows["manual"][1]["ssh_connection"] == {
        "hostname": "gpu.example.test",
        "port": 2222,
        "user": "trainer",
        "identity_file": "~/.ssh/t",
    }


def test_migration_downgrade_restores_columns(tmp_path: Path) -> None:
    alembic_cfg, db_path = _upgrade_to_pre_migration(tmp_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO remote_trainers (id, name, connection_mode, url, ssh_host_alias, ssh_remote_port, "
            "ssh_local_port) VALUES ('alias', 'alias trainer', 'ssh', 'http://127.0.0.1:9001', 'gpu-box', 8001, 9001)"
        )
        connection.commit()

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)
    command.downgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT url, ssh_host_alias, ssh_remote_port, ssh_local_port FROM remote_trainers WHERE id = 'alias'"
        ).fetchone()

    assert row == ("http://127.0.0.1:9001", "gpu-box", 8001, 9001)
