# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Action Dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from action_trainer.data import Observation


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
    def features(self) -> dict:
        """Raw dataset features."""

    @property
    @abstractmethod
    def action_features(self) -> dict:
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
