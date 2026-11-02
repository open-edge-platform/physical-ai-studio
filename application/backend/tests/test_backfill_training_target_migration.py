"""Functional test for the `1510153d39a2` data migration.

Runs the actual Alembic upgrade against a throwaway SQLite database seeded
with legacy training job payloads (no `training_target` key at all, as
persisted before the field existed), then asserts the migration backfills a
payload that validates cleanly through the `Job` schema.
"""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from alembic import command
from db.migration import MigrationManager
from settings import Settings

_PRE_MIGRATION_REVISION = "b7a4c1e9f0d2"
_MIGRATION_REVISION = "1510153d39a2"


def _seed_legacy_training_job(db_path: Path, *, payload: dict) -> str:
    job_id = str(uuid4())
    project_id = str(uuid4())
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "INSERT INTO projects (id, name) VALUES (?, ?)",
            (project_id, "Test project"),
        )
        connection.execute(
            "INSERT INTO jobs (id, project_id, type, progress, status, message, payload) "
            "VALUES (?, ?, 'training', 0, 'pending', 'Job created', ?)",
            (job_id, project_id, json.dumps(payload)),
        )
        connection.commit()
    finally:
        connection.close()
    return job_id


def _read_payload(db_path: Path, job_id: str) -> dict:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        connection.close()
    assert row is not None
    return json.loads(row[0])


def test_migration_backfills_local_target_when_no_remote_trainer_id(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    manager = MigrationManager(settings)
    alembic_cfg = manager.get_alembic_config()

    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    db_path = settings.data_dir / settings.database_file
    job_id = _seed_legacy_training_job(
        db_path,
        payload={
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "legacy-local-model",
            "batch_size": 8,
        },
    )

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    payload = _read_payload(db_path, job_id)
    assert payload["training_target"] == "local"
    assert "remote_trainer_id" not in payload


def test_migration_backfills_remote_target_when_remote_trainer_id_present(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    manager = MigrationManager(settings)
    alembic_cfg = manager.get_alembic_config()

    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    db_path = settings.data_dir / settings.database_file
    remote_trainer_id = str(uuid4())
    job_id = _seed_legacy_training_job(
        db_path,
        payload={
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "legacy-remote-model",
            "batch_size": 8,
            "remote_trainer_id": remote_trainer_id,
        },
    )

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    payload = _read_payload(db_path, job_id)
    assert payload["training_target"] == "remote"
    assert payload["remote_trainer_id"] == remote_trainer_id


def test_migration_drops_stale_remote_fields_from_local_payload(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    manager = MigrationManager(settings)
    alembic_cfg = manager.get_alembic_config()

    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    db_path = settings.data_dir / settings.database_file
    job_id = _seed_legacy_training_job(
        db_path,
        payload={
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "act",
            "model_name": "legacy-local-with-stale-fields",
            "batch_size": 8,
            "training_target": "local",
            "remote_trainer_id": None,
            "remote_trainer_url": None,
            "remote_trainer_name": None,
        },
    )

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    payload = _read_payload(db_path, job_id)
    assert payload["training_target"] == "local"
    assert "remote_trainer_id" not in payload
    assert "remote_trainer_url" not in payload
    assert "remote_trainer_name" not in payload


def test_migration_is_a_noop_for_up_to_date_payloads(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    manager = MigrationManager(settings)
    alembic_cfg = manager.get_alembic_config()

    command.upgrade(alembic_cfg, _PRE_MIGRATION_REVISION)

    db_path = settings.data_dir / settings.database_file
    original_payload = {
        "project_id": str(uuid4()),
        "dataset_id": str(uuid4()),
        "policy": "act",
        "model_name": "modern-remote-model",
        "batch_size": 8,
        "training_target": "remote",
        "remote_trainer_id": str(uuid4()),
    }
    job_id = _seed_legacy_training_job(db_path, payload=original_payload)

    command.upgrade(alembic_cfg, _MIGRATION_REVISION)

    assert _read_payload(db_path, job_id) == original_payload
