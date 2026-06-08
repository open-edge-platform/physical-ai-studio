from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import storage_migration
from db.schema import (
    Base,
    DatasetDB,
    ModelDB,
    ProjectDB,
    ProjectEnvironmentDB,
    ProjectRobotDB,
    RobotCalibrationDB,
    SnapshotDB,
)
from schemas.robot import RobotType
from settings import Settings
from storage_migration import StorageMigrationError, migrate_default_storage_dir


def _settings(tmp_path: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "new-storage"))


def _old_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    old_storage = tmp_path / "old-cache" / "physicalai"
    monkeypatch.setattr(storage_migration, "OLD_DEFAULT_STORAGE_DIR", old_storage)
    return old_storage


def test_migration_noops_when_old_storage_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    _old_storage(tmp_path, monkeypatch)

    migrate_default_storage_dir(settings, interactive=False)

    assert not settings.storage_dir.exists()


def test_migration_skips_explicit_storage_dir_without_auto_migrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    old_storage.mkdir(parents=True)
    (old_storage / "datasets").mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(settings.storage_dir))
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)

    migrate_default_storage_dir(settings, interactive=False)

    assert old_storage.exists()
    assert not settings.storage_dir.exists()


def test_migration_moves_old_storage_to_missing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets").mkdir(parents=True)
    (old_storage / "datasets" / "dataset.txt").write_text("data")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert not old_storage.exists()
    assert (settings.storage_dir / "datasets" / "dataset.txt").read_text() == "data"


def test_migration_replaces_empty_storage_scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "models").mkdir(parents=True)
    (old_storage / "models" / "model.txt").write_text("model")
    for subdir in storage_migration.EXPECTED_STORAGE_SUBDIRS:
        (settings.storage_dir / subdir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    migrate_default_storage_dir(settings, interactive=False)

    assert (settings.storage_dir / "models" / "model.txt").read_text() == "model"


def test_migration_fails_when_destination_contains_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    old_storage.mkdir(parents=True)
    settings.storage_dir.mkdir(parents=True)
    (settings.storage_dir / "existing.txt").write_text("data")
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    with pytest.raises(StorageMigrationError, match="already contains files"):
        migrate_default_storage_dir(settings, interactive=False)

    assert old_storage.exists()


def test_migration_fails_non_interactive_without_auto_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    old_storage.mkdir(parents=True)
    monkeypatch.delenv("AUTO_MIGRATE_STORAGE_DIR", raising=False)

    with pytest.raises(StorageMigrationError, match="non-interactive"):
        migrate_default_storage_dir(settings, interactive=False)

    assert old_storage.exists()


def test_migration_rewrites_database_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path)
    old_storage = _old_storage(tmp_path, monkeypatch)
    (old_storage / "datasets" / "dataset-1").mkdir(parents=True)
    settings.data_dir.mkdir(parents=True)
    monkeypatch.setenv("AUTO_MIGRATE_STORAGE_DIR", "true")

    engine = create_engine(settings.database_url_sync)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        session.add_all(
            [
                ProjectDB(id="project-1", name="Project"),
                ProjectEnvironmentDB(id="environment-1", project_id="project-1", name="Environment"),
                ProjectRobotDB(
                    id="robot-1",
                    project_id="project-1",
                    name="Robot",
                    type=RobotType.SO101_FOLLOWER,
                    payload={},
                ),
            ]
        )
        session.commit()

        session.add_all(
            [
                DatasetDB(
                    id="dataset-1",
                    name="Dataset",
                    path=str(old_storage / "datasets" / "dataset-1"),
                    project_id="project-1",
                    environment_id="environment-1",
                    default_task="",
                ),
                DatasetDB(
                    id="dataset-2",
                    name="External Dataset",
                    path=str(tmp_path / "external" / "dataset-2"),
                    project_id="project-1",
                    environment_id="environment-1",
                    default_task="",
                ),
                ModelDB(
                    id="model-1",
                    name="Model",
                    path=str(old_storage / "models" / "model-1"),
                    policy="act",
                    properties={},
                    project_id="project-1",
                ),
                SnapshotDB(
                    id="snapshot-1",
                    path=str(old_storage / "snapshots" / "snapshot-1"),
                    dataset_id="dataset-1",
                ),
                RobotCalibrationDB(
                    id="calibration-1",
                    file_path=str(old_storage / "robots" / "robot-1" / "calibration.json"),
                    robot_id="robot-1",
                ),
            ]
        )
        session.commit()

    migrate_default_storage_dir(settings, interactive=False)

    with session_factory() as session:
        dataset_path = session.scalar(select(DatasetDB.path).where(DatasetDB.id == "dataset-1"))
        external_dataset_path = session.scalar(select(DatasetDB.path).where(DatasetDB.id == "dataset-2"))
        model_path = session.scalar(select(ModelDB.path).where(ModelDB.id == "model-1"))
        snapshot_path = session.scalar(select(SnapshotDB.path).where(SnapshotDB.id == "snapshot-1"))
        calibration_path = session.scalar(
            select(RobotCalibrationDB.file_path).where(RobotCalibrationDB.id == "calibration-1")
        )

    assert dataset_path == str(settings.storage_dir / "datasets" / "dataset-1")
    assert external_dataset_path == str(tmp_path / "external" / "dataset-2")
    assert model_path == str(settings.storage_dir / "models" / "model-1")
    assert snapshot_path == str(settings.storage_dir / "snapshots" / "snapshot-1")
    assert calibration_path == str(settings.storage_dir / "robots" / "robot-1" / "calibration.json")
    engine.dispose()
