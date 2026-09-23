"""Verify the migration removes unsupported SSH jobs and keeps other records."""

import json
import sqlite3
from pathlib import Path

import pytest

from alembic import command
from db.migration import MigrationManager
from settings import Settings


@pytest.mark.parametrize("container_column", ["container_name", "container_id"])
def test_migration_refuses_to_orphan_provisioned_containers(tmp_path: Path, container_column: str) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    alembic_cfg = MigrationManager(settings).get_alembic_config()
    command.upgrade(alembic_cfg, "4b8d2f6a1c30")
    db_path = settings.data_dir / settings.database_file
    with sqlite3.connect(db_path) as connection:
        connection.execute("INSERT INTO projects (id, name) VALUES ('project', 'Test project')")
        connection.execute(
            "INSERT INTO remote_servers (id, name, ssh_host_alias, device_type) "
            "VALUES ('server', 'GPU', 'gpu-host', 'cuda')"
        )
        connection.execute(
            "INSERT INTO jobs (id, project_id, type, progress, status, message, payload) "
            "VALUES ('ssh-job', 'project', 'training', 0, 'running', 'Working', ?)",
            (json.dumps({"training_target": "ssh"}),),
        )
        connection.execute(
            "INSERT INTO job_provisioning "
            "(job_id, remote_server_id, ssh_host_alias, container_name, container_id) "
            "VALUES ('ssh-job', 'server', 'gpu-host', ?, ?)",
            (
                "physicalai-trainer-ssh-job" if container_column == "container_name" else None,
                "physicalai-trainer-ssh-job" if container_column == "container_id" else None,
            ),
        )

    with pytest.raises(RuntimeError, match="physicalai-trainer-ssh-job"):
        command.upgrade(alembic_cfg, "7c1a9e2b4d6f")
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT coalesce(container_name, container_id) FROM job_provisioning").fetchone() == (
            "physicalai-trainer-ssh-job",
        )
        connection.execute("DELETE FROM job_provisioning WHERE job_id = 'ssh-job'")

    command.upgrade(alembic_cfg, "7c1a9e2b4d6f")


def test_migration_removes_only_unsupported_ssh_jobs(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    alembic_cfg = MigrationManager(settings).get_alembic_config()
    command.upgrade(alembic_cfg, "4b8d2f6a1c30")

    db_path = settings.data_dir / settings.database_file
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("INSERT INTO projects (id, name) VALUES ('project', 'Test project')")
        connection.execute(
            "INSERT INTO remote_servers (id, name, ssh_host_alias, device_type) "
            "VALUES ('server', 'GPU', 'gpu-host', 'cuda')"
        )
        for job_id, target in (("ssh-job", "ssh"), ("local-job", "local")):
            connection.execute(
                "INSERT INTO jobs (id, project_id, type, progress, status, message, payload) "
                "VALUES (?, 'project', 'training', 0, 'completed', 'Done', ?)",
                (job_id, json.dumps({"training_target": target})),
            )
        connection.execute(
            "INSERT INTO job_provisioning (job_id, remote_server_id, ssh_host_alias) "
            "VALUES ('ssh-job', 'server', 'gpu-host')"
        )
        connection.execute(
            "INSERT INTO models (id, name, path, policy, properties, project_id, train_job_id) "
            "VALUES ('model', 'Model', 'model.pt', 'act', '{}', 'project', 'ssh-job')"
        )

    command.upgrade(alembic_cfg, "7c1a9e2b4d6f")

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        assert connection.execute("SELECT id FROM jobs").fetchall() == [("local-job",)]
        assert connection.execute("SELECT id, train_job_id FROM models").fetchall() == [("model", None)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert not {"remote_servers", "job_provisioning"} & {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
