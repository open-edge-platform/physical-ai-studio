# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobotDataModule argument forwarding."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from physicalai.data.lerobot import dataset as lerobot_dataset_module
from physicalai.data.lerobot.datamodule import LeRobotDataModule


def test_datamodule_forwards_default_transform_flag() -> None:
    """DataModule should forward default transform flag to adapter."""
    with patch("physicalai.data.lerobot.datamodule._LeRobotDatasetAdapter") as mock_adapter:
        mock_adapter.return_value = object()
        LeRobotDataModule(
            repo_id="any/repo",
            train_batch_size=1,
            use_default_image_transforms=True,
            data_format="physicalai",
        )

    called_kwargs = mock_adapter.call_args.kwargs
    assert "use_default_image_transforms" in called_kwargs  # noqa: S101
    assert called_kwargs["use_default_image_transforms"] is True  # noqa: S101


def test_resolve_image_transforms_builds_default_pipeline() -> None:
    """Default transform resolver should create LeRobot transform pipeline."""

    @dataclass
    class FakeImageTransformsConfig:
        enable: bool

    @dataclass
    class FakeImageTransforms:
        config: FakeImageTransformsConfig

    with (
        patch.object(lerobot_dataset_module, "ImageTransforms", FakeImageTransforms),
        patch.object(lerobot_dataset_module, "ImageTransformsConfig", FakeImageTransformsConfig),
    ):
        transforms = lerobot_dataset_module._resolve_image_transforms(  # noqa: SLF001
            None,
            use_default_image_transforms=True,
        )

    assert isinstance(transforms, FakeImageTransforms)  # noqa: S101
    assert transforms.config.enable is True  # noqa: S101
