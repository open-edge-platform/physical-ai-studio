# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer data APIs, loaded on demand so archive helpers stay lightweight."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .datamodules import DataModule
    from .dataset import Dataset
    from .lerobot import LeRobotDataModule
    from .observation import Feature, FeatureType, NormalizationParameters, NormalizationValue, Observation

_LAZY_EXPORTS = {
    "DataModule": "datamodules",
    "Dataset": "dataset",
    "LeRobotDataModule": "lerobot",
    "Feature": "observation",
    "FeatureType": "observation",
    "NormalizationParameters": "observation",
    "NormalizationValue": "observation",
    "Observation": "observation",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load a public data export only when it is requested.

    Returns:
        The requested data class or type.

    Raises:
        AttributeError: If the export is unknown.
    """
    if name not in _LAZY_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "DataModule",
    "Dataset",
    "Feature",
    "FeatureType",
    "LeRobotDataModule",
    "NormalizationParameters",
    "NormalizationValue",
    "Observation",
]
