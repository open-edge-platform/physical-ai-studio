from unittest.mock import patch
from uuid import uuid4

from internal_datasets.access_mode import DatasetAccessMode
from internal_datasets.utils import get_internal_read_dataset, get_internal_recording_dataset
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


def test_get_internal_recording_dataset_uses_recording_mode() -> None:
    dataset = _make_dataset()
    with patch("internal_datasets.utils.InternalLeRobotDataset") as mocked_cls:
        get_internal_recording_dataset(dataset)

    mocked_cls.assert_called_once()
    _, kwargs = mocked_cls.call_args
    assert kwargs["access_mode"] is DatasetAccessMode.RECORDING_MUTATION
