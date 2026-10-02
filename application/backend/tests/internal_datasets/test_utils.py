import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.utils import _load_read_dataset, get_internal_read_dataset, get_internal_recording_dataset
from schemas.dataset import Dataset


def _make_dataset() -> Dataset:
    return Dataset.model_validate(
        {
            "id": str(uuid4()),
            "name": "dataset",
            "path": "/tmp/dataset",
            "default_task": "task",
            "project_id": str(uuid4()),
            "environment_id": str(uuid4()),
        }
    )


def test_get_internal_read_dataset_uses_read_only_mode() -> None:
    dataset = _make_dataset()
    with patch("internal_datasets.utils.InternalLeRobotDataset") as mocked_cls:
        get_internal_read_dataset(dataset)

    mocked_cls.assert_called_once()
    _, kwargs = mocked_cls.call_args
    assert kwargs["access_mode"] is DatasetAccessMode.READ_ONLY


def test_read_dataset_cache_refreshes_when_metadata_changes(tmp_path: Path) -> None:
    info = tmp_path / "meta/info.json"
    info.parent.mkdir()
    info.write_text("{}")
    dataset = _make_dataset()
    _load_read_dataset.cache_clear()
    with (
        patch.object(Dataset, "path", property(lambda _: str(tmp_path))),
        patch("internal_datasets.utils.InternalLeRobotDataset") as mocked_cls,
    ):
        first = get_internal_read_dataset(dataset)
        assert get_internal_read_dataset(dataset) is first
        assert mocked_cls.call_count == 1

        stat = info.stat()
        os.utime(info, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        get_internal_read_dataset(dataset)
        assert mocked_cls.call_count == 2
    _load_read_dataset.cache_clear()


def test_get_internal_recording_dataset_uses_recording_mode() -> None:
    dataset = _make_dataset()
    with patch("internal_datasets.utils.InternalLeRobotDataset") as mocked_cls:
        get_internal_recording_dataset(dataset)

    mocked_cls.assert_called_once()
    _, kwargs = mocked_cls.call_args
    assert kwargs["access_mode"] is DatasetAccessMode.RECORDING_MUTATION
