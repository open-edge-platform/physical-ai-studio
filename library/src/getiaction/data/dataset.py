# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Action Dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from getiaction.data import Feature, Observation


class Dataset(TorchDataset, ABC):
    """An abstract base class for datasets that return observations."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Observation:
        """Loads and returns an Observation at the given index."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of Observations in the dataset."""

    @property
    @abstractmethod
    def raw_features(self) -> dict:
        """Raw dataset features."""

    @property
    @abstractmethod
    def observation_features(self) -> dict[str, Feature]:
        """Observation features from the dataset."""

    @property
    @abstractmethod
    def action_features(self) -> dict[str, Feature]:
        """Action features from the dataset."""

    @property
    @abstractmethod
    def fps(self) -> int:
        """Frames per second of the dataset."""

    @property
    @abstractmethod
    def tolerance_s(self) -> float:
        """Tolerance to keep delta timestamps in sync with fps."""

    @property
    @abstractmethod
    def delta_indices(self) -> dict[str, list[int]]:
        """Exposes delta_indices from the dataset."""

    @delta_indices.setter
    @abstractmethod
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Allows setting delta_indices on the dataset."""

    @property
    def stats(self) -> dict[str, dict[str, list[float]]]:
        """Normalization statistics extracted from features.

        Returns:
            Dict mapping feature keys to their normalization stats
            (mean, std, min, max). Keys follow the format used by
            the underlying dataset (e.g., "observation.state", "action").
        """
        stats_dict: dict[str, dict[str, list[float]]] = {}

        for name, feature in self.observation_features.items():
            if feature.normalization_data is not None:
                norm = feature.normalization_data
                stats_dict[f"observation.{name}"] = {
                    stat: list(val) if hasattr(val, "__iter__") else [val]
                    for stat in ("mean", "std", "min", "max")
                    if (val := getattr(norm, stat, None)) is not None
                }

        for name, feature in self.action_features.items():
            if feature.normalization_data is not None:
                norm = feature.normalization_data
                stats_dict[name] = {
                    stat: list(val) if hasattr(val, "__iter__") else [val]
                    for stat in ("mean", "std", "min", "max")
                    if (val := getattr(norm, stat, None)) is not None
                }

        return stats_dict
