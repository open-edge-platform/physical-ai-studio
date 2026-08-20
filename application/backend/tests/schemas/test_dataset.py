from pathlib import Path
from uuid import uuid4

import pytest

from schemas import Dataset
from settings import Settings


def _point_datasets_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Point Dataset.path's computed location at tmp_path."""
    settings = Settings(DATA_DIR=str(tmp_path / "data"), STORAGE_DIR=str(tmp_path / "storage"))
    monkeypatch.setattr("schemas.dataset.get_settings", lambda: settings)
    return settings


def test_path_is_derived_from_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _point_datasets_dir(monkeypatch, tmp_path)
    dataset = Dataset(name="x", default_task="", project_id=uuid4(), environment_id=uuid4())

    assert Path(dataset.path) == settings.datasets_dir / str(dataset.id)


def test_same_name_in_different_projects_get_distinct_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _point_datasets_dir(monkeypatch, tmp_path)
    environment_id = uuid4()
    first = Dataset(name="Collect blocks", default_task="", project_id=uuid4(), environment_id=environment_id)
    second = Dataset(name="Collect blocks", default_task="", project_id=uuid4(), environment_id=environment_id)

    assert first.path != second.path


def test_client_supplied_path_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _point_datasets_dir(monkeypatch, tmp_path)
    dataset = Dataset.model_validate(
        {
            "name": "x",
            "default_task": "",
            "project_id": str(uuid4()),
            "environment_id": str(uuid4()),
            "path": "/attacker/controlled/path",
        }
    )

    assert Path(dataset.path) == settings.datasets_dir / str(dataset.id)


def test_path_is_excluded_from_serialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _point_datasets_dir(monkeypatch, tmp_path)
    dataset = Dataset(name="x", default_task="", project_id=uuid4(), environment_id=uuid4())

    # path is internal-only: accessible as an attribute, absent from API responses.
    assert dataset.path
    assert "path" not in dataset.model_dump()
    assert "path" not in dataset.model_dump(mode="json")
