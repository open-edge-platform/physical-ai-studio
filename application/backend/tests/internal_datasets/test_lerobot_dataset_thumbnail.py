from unittest.mock import MagicMock

import numpy as np

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset


def test_thumbnail_reader_fallback_on_recording_runtime_error() -> None:
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)
    reader = MagicMock()
    reader.hf_dataset = object()
    reader.get_item.return_value = {"observation.images.main": MagicMock()}
    reader.get_item.return_value[
        "observation.images.main"
    ].permute.return_value.detach.return_value.numpy.return_value = np.zeros((8, 8, 3), dtype=np.float32)

    def _getitem(_index: int):
        raise RuntimeError(
            "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
        )

    dataset._dataset = MagicMock()
    dataset._dataset.__getitem__.side_effect = _getitem
    dataset._dataset.reader = reader

    result = dataset._read_dataset_item_for_thumbnail(0)

    assert result is not None
    reader.get_item.assert_called_once_with(0)


def test_thumbnail_reader_fallback_activates_reader_when_needed() -> None:
    dataset = InternalLeRobotDataset.__new__(InternalLeRobotDataset)

    reader = MagicMock()
    reader.hf_dataset = None
    reader.get_item.return_value = {"observation.images.main": MagicMock()}

    def _getitem(_index: int):
        raise RuntimeError(
            "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
        )

    dataset._dataset = MagicMock()
    dataset._dataset.__getitem__.side_effect = _getitem
    dataset._dataset.reader = reader

    result = dataset._read_dataset_item_for_thumbnail(0)

    assert result is not None
    reader.load_and_activate.assert_called_once()
