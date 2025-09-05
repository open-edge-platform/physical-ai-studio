# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base Action Dataset
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from action_trainer.data.types import Observation


class ActionDataset(Dataset, ABC):
    """
    An abstract base class for datasets that return observations.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Observation:
        """Loads and returns an Observation at the given index."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of Observations in the dataset."""
