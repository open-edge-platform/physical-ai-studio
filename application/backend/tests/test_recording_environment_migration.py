"""Regression tests for rolling back dataset recording environments."""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from alembic import command
from db.migration import MigrationManager
from db.schema import DatasetDB, DatasetRecordingEnvironmentDB, ModelDB, ProjectDB, ProjectEnvironmentDB, SnapshotDB
from settings import Settings


def test_downgrade_restores_environment_fk_and_preserves_dependents(tmp_path: Path) -> None:
    settings = Settings(STORAGE_DIR=tmp_path, DATABASE_FILE="test.db")
    config = MigrationManager(settings).get_alembic_config()
    command.upgrade(config, "e5f6a7b8c9d0")

    engine = create_engine(settings.database_url_sync)
    with Session(engine) as session:
        session.add(ProjectDB(id="project", name="Project"))
        session.flush()
        session.add(ProjectEnvironmentDB(id="environment", project_id="project", name="Environment"))
        session.add(DatasetDB(id="dataset", name="Dataset", path="/dataset", project_id="project", default_task=""))
        session.flush()
        session.add(DatasetRecordingEnvironmentDB(id="association", dataset_id="dataset", environment_id="environment"))
        session.add(SnapshotDB(id="snapshot", path="/snapshot", dataset_id="dataset"))
        session.flush()
        session.add(
            ModelDB(
                id="model",
                name="Model",
                path="/model",
                policy="act",
                properties={},
                dataset_id="dataset",
                project_id="project",
                snapshot_id="snapshot",
            )
        )
        session.commit()
    engine.dispose()

    command.downgrade(config, "91d5e3a7c2b0")

    with sqlite3.connect(settings.data_dir / settings.database_file) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        assert connection.execute("SELECT environment_id FROM datasets WHERE id = 'dataset'").fetchone() == (
            "environment",
        )
        assert connection.execute("SELECT dataset_id FROM snapshots WHERE id = 'snapshot'").fetchone() == ("dataset",)
        assert connection.execute("SELECT dataset_id, snapshot_id FROM models WHERE id = 'model'").fetchone() == (
            "dataset",
            "snapshot",
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            connection.execute("DELETE FROM project_environments WHERE id = 'environment'")
