# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer datamodules."""

from typing import Any

# Import torch-free constants first
from .constants import (
    ACTION,
    EPISODE_INDEX,
    EXTRA,
    FRAME_INDEX,
    IMAGES,
    INDEX,
    INFO,
    NEXT_REWARD,
    NEXT_SUCCESS,
    STATE,
    TASK,
    TASK_INDEX,
    TIMESTAMP,
)


# Lazy-load torch-dependent modules to keep constants torch-free
def _get_lazy_attr(module: str, name: str) -> Any:  # noqa: ANN401
    """Lazy load a module attribute.

    Args:
        module: Module name to import from (e.g., '.datamodules' for relative or 'pkg.mod' for absolute).
        name: Attribute name to retrieve.

    Returns:
        The requested attribute.
    """
    if module.startswith("."):
        # Relative import - convert to absolute using current package
        module = f"physicalai.data{module}"
    return getattr(__import__(module, fromlist=[name]), name)


__torch_dependent_attrs = {
    "DataModule": lambda: _get_lazy_attr(".datamodules", "DataModule"),
    "Dataset": lambda: _get_lazy_attr(".dataset", "Dataset"),
    "LeRobotDataModule": lambda: _get_lazy_attr(".lerobot", "LeRobotDataModule"),
    "Feature": lambda: _get_lazy_attr(".observation", "Feature"),
    "FeatureType": lambda: _get_lazy_attr(".observation", "FeatureType"),
    "NormalizationParameters": lambda: _get_lazy_attr(".observation", "NormalizationParameters"),
    "Observation": lambda: _get_lazy_attr(".observation", "Observation"),
}


def __getattr__(name: str):  # noqa: ANN202
    """Lazy load torch-dependent modules to keep constants torch-free.

    Args:
        name: Name of attribute to load.

    Returns:
        The requested module attribute.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name in __torch_dependent_attrs:
        return __torch_dependent_attrs[name]()
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


__all__ = [
    "ACTION",
    "EPISODE_INDEX",
    "EXTRA",
    "FRAME_INDEX",
    "IMAGES",
    "INDEX",
    "INFO",
    "NEXT_REWARD",
    "NEXT_SUCCESS",
    "STATE",
    "TASK",
    "TASK_INDEX",
    "TIMESTAMP",
    "DataModule",  # noqa: F822
    "Dataset",  # noqa: F822
    "Feature",  # noqa: F822
    "FeatureType",  # noqa: F822
    "LeRobotDataModule",  # noqa: F822
    "NormalizationParameters",  # noqa: F822
    "Observation",  # noqa: F822
]
