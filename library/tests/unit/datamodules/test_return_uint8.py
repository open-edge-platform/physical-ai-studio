# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Keep decoded RGB frames in uint8 only when requested."""

from unittest.mock import patch

import pytest
import torch

from physicalai.data import Observation
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter


class FakeLeRobotDataset:
    def __init__(self, *, return_uint8: bool = False, **kwargs):
        self.return_uint8 = return_uint8

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        pixels = torch.arange(24, dtype=torch.uint8).reshape(3, 2, 4)
        return {
            "observation.images.front": pixels if self.return_uint8 else pixels.float() / 255,
            "action": torch.zeros(2),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(idx / 30),
        }


@pytest.mark.parametrize("return_uint8", [False, True])
def test_adapter_preserves_decoded_image_dtype(return_uint8):
    with patch("physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset):
        dataset = _LeRobotDatasetAdapter(repo_id="test/robot", return_uint8=return_uint8)
    pixels = dataset[0].images["front"]
    assert pixels.dtype == (torch.uint8 if return_uint8 else torch.float32)
    expected = torch.arange(24, dtype=torch.uint8).reshape(3, 2, 4)
    torch.testing.assert_close(pixels if return_uint8 else pixels * 255, expected.to(pixels.dtype))


@pytest.mark.parametrize("data_format", ["physicalai", "lerobot"])
@pytest.mark.parametrize("return_uint8", [False, True])
def test_datamodule_forwards_dtype_to_train_and_validation(data_format, return_uint8):
    with (
        patch("physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset),
        patch("physicalai.data.lerobot.datamodule.LeRobotDataset", FakeLeRobotDataset),
        patch("physicalai.data.lerobot.datamodule._read_total_episodes", return_value=20),
        patch.object(LeRobotDataModule, "_ensure_quantile_stats"),
    ):
        dm = LeRobotDataModule(
            repo_id="test/robot",
            return_uint8=return_uint8,
            data_format=data_format,
            val_split=0.1 if data_format == "physicalai" else 0,
            num_workers=0,
            train_batch_size=2,
        )
    batch = next(iter(dm.train_dataloader()))
    pixels = batch.images["front"] if isinstance(batch, Observation) else batch["observation.images.front"]
    expected_dtype = torch.uint8 if return_uint8 else torch.float32
    assert pixels.dtype == expected_dtype
    if data_format == "physicalai":
        assert dm.val_eval_dataset[0].images["front"].dtype == expected_dtype


def test_float_frames_remain_the_default():
    with patch("physicalai.data.lerobot.dataset.LeRobotDataset", FakeLeRobotDataset):
        dataset = _LeRobotDatasetAdapter(repo_id="test/robot")
    assert dataset[0].images["front"].dtype == torch.float32
