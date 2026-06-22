import importlib.util
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from db.schema import Base, DatasetDB, ProjectDB, ProjectEnvironmentDB
from settings import Settings

# The Alembic revision owns the relocation logic. Its directory shares the name of
# the installed ``alembic`` package and lacks an ``__init__.py``, so load it by path.
_revision_path = (
    Path(__file__).resolve().parents[1] / "src" / "alembic" / "versions" / "f3a1c9d2b8e4_dataset_id_based_folders.py"
)
_spec = importlib.util.spec_from_file_location("dataset_id_based_folders_migration", _revision_path)
assert _spec is not None and _spec.loader is not None
_migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_migration)
relocate_dataset_paths = _migration.relocate_dataset_paths


def _settings(tmp_path: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))


def _migrate(settings: Settings, *, dry_run: bool = False) -> int:
    """Run the relocation helper in a transaction, mirroring the Alembic upgrade."""
    engine = create_engine(settings.database_url_sync, connect_args={"check_same_thread": False})
    with engine.begin() as connection:
        migrated = relocate_dataset_paths(connection, settings.datasets_dir.expanduser().resolve(), dry_run=dry_run)
    engine.dispose()
    return migrated


def _seed(settings: Settings, datasets: list[tuple[str, str]]) -> None:
    """Create the database with one project/environment and the given (id, path) datasets."""
    (settings.data_dir).mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url_sync, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        session.add_all(
            [
                ProjectDB(id="project-1", name="Project"),
                ProjectEnvironmentDB(id="environment-1", project_id="project-1", name="Environment"),
            ]
        )
        session.commit()
        session.add_all(
            [
                DatasetDB(
                    id=dataset_id,
                    name=dataset_id,
                    path=path,
                    project_id="project-1",
                    environment_id="environment-1",
                    default_task="",
                )
                for dataset_id, path in datasets
            ]
        )
        session.commit()
    engine.dispose()


def _read_path(settings: Settings, dataset_id: str) -> str:
    engine = create_engine(settings.database_url_sync, connect_args={"check_same_thread": False})
    with sessionmaker(bind=engine)() as session:
        path = session.scalar(select(DatasetDB.path).where(DatasetDB.id == dataset_id))
    engine.dispose()
    assert path is not None
    return path


def test_shared_folder_is_split_into_independent_copies(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    shared = settings.datasets_dir / "collect_blocks"
    shared.mkdir(parents=True)
    (shared / "episode.txt").write_text("episode")

    _seed(settings, [("dataset-a", str(shared)), ("dataset-b", str(shared))])

    migrated = _migrate(settings)

    assert migrated == 2
    path_a = Path(_read_path(settings, "dataset-a"))
    path_b = Path(_read_path(settings, "dataset-b"))
    assert path_a == settings.datasets_dir / "dataset-a"
    assert path_b == settings.datasets_dir / "dataset-b"
    # Each dataset now owns an independent copy of the episodes.
    assert (path_a / "episode.txt").read_text() == "episode"
    assert (path_b / "episode.txt").read_text() == "episode"


def test_already_id_based_path_is_untouched(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    target = settings.datasets_dir / "dataset-a"
    target.mkdir(parents=True)
    (target / "episode.txt").write_text("episode")

    _seed(settings, [("dataset-a", str(target))])

    migrated = _migrate(settings)

    assert migrated == 0
    assert Path(_read_path(settings, "dataset-a")) == target


def test_dry_run_reports_without_changing_anything(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = settings.datasets_dir / "old_name"
    source.mkdir(parents=True)

    _seed(settings, [("dataset-a", str(source))])

    migrated = _migrate(settings, dry_run=True)

    assert migrated == 1
    # Database and filesystem are unchanged.
    assert Path(_read_path(settings, "dataset-a")) == source
    assert source.exists()
    assert not (settings.datasets_dir / "dataset-a").exists()


def test_single_owner_folder_is_moved(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = settings.datasets_dir / "old_name"
    source.mkdir(parents=True)
    (source / "episode.txt").write_text("episode")

    _seed(settings, [("dataset-a", str(source))])

    migrated = _migrate(settings)

    assert migrated == 1
    target = settings.datasets_dir / "dataset-a"
    assert (target / "episode.txt").read_text() == "episode"
    assert not source.exists()  # Single owner is moved, not copied.
    assert Path(_read_path(settings, "dataset-a")) == target


def test_missing_source_folder_creates_empty_target(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    missing = settings.datasets_dir / "gone"

    _seed(settings, [("dataset-a", str(missing))])

    migrated = _migrate(settings)

    assert migrated == 1
    target = settings.datasets_dir / "dataset-a"
    assert target.is_dir()
    assert Path(_read_path(settings, "dataset-a")) == target
