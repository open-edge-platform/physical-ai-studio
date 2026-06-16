from pathlib import Path
from uuid import uuid4

import pytest

from db.schema import DatasetDB
from repositories.mappers import DatasetMapper
from schemas import Dataset
from settings import Settings


def _point_datasets_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    settings = Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))
    monkeypatch.setattr("schemas.dataset.get_settings", lambda: settings)
    return settings


def test_to_schema_persists_computed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _point_datasets_dir(monkeypatch, tmp_path)
    dataset = Dataset(name="x", default_task="t", project_id=uuid4(), environment_id=uuid4())

    db_row = DatasetMapper.to_schema(dataset)

    assert db_row.path == str(settings.datasets_dir / str(dataset.id))


def test_from_schema_ignores_stored_path_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The DB column is a write-through cache; path is always recomputed from the id on read.
    settings = _point_datasets_dir(monkeypatch, tmp_path)
    db_row = DatasetDB(
        id=str(uuid4()),
        name="x",
        path="/legacy/name-based/location",
        default_task="t",
        project_id=str(uuid4()),
        environment_id=str(uuid4()),
    )

    dataset = DatasetMapper.from_schema(db_row)

    assert Path(dataset.path) == settings.datasets_dir / db_row.id
    assert dataset.path != "/legacy/name-based/location"
