# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset integration for PhysicalAI.

This package provides integration with HuggingFace LeRobot datasets, including:
- Format conversion between PhysicalAI Observation and LeRobot dict formats
- Dataset adapter for wrapping LeRobotDataset
- DataModule for PyTorch Lightning integration
- Utilities for delta timestamps configuration

Examples:
    >>> from physicalai.data.lerobot import LeRobotDataModule, FormatConverter

    >>> # Create datamodule
    >>> datamodule = LeRobotDataModule(
    ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
    ...     train_batch_size=32,
    ...     data_format="lerobot"
    ... )

    >>> # Convert between observation dict formats
    >>> lerobot_dict = FormatConverter.to_lerobot_dict(observation)
    >>> observation = FormatConverter.to_observation(lerobot_dict)

    >>> # Get delta timestamps for a policy
    >>> from physicalai.data.lerobot import get_delta_timestamps_from_policy
    >>> delta_timestamps = get_delta_timestamps_from_policy("act", fps=10)
"""

from .converters import DataFormat, FormatConverter
from .datamodule import LeRobotDataModule
from .utils import get_delta_timestamps_from_policy

__all__ = ["DataFormat", "FormatConverter", "LeRobotDataModule", "get_delta_timestamps_from_policy"]
