# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset integration for GetiAction.

This package provides integration with HuggingFace LeRobot datasets, including:
- Format conversion between GetiAction Observation and LeRobot dict formats
- Dataset adapter for wrapping LeRobotDataset
- DataModule for PyTorch Lightning integration

Example:
    >>> from getiaction.data.lerobot import LeRobotDataModule, FormatConverter
    >>>
    >>> # Create datamodule
    >>> datamodule = LeRobotDataModule(
    ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
    ...     train_batch_size=32,
    ...     data_format="lerobot"
    ... )
    >>>
    >>> # Convert between formats
    >>> lerobot_dict = FormatConverter.to_lerobot_dict(observation)
    >>> observation = FormatConverter.to_observation(lerobot_dict)
"""

from .converters import DataFormat, FormatConverter
from .datamodule import LeRobotDataModule

__all__ = ["DataFormat", "FormatConverter", "LeRobotDataModule"]
