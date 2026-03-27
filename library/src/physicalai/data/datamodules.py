# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Lightning datamodules."""

from __future__ import annotations

import logging
import os
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset as TorchDataset, TensorDataset

from physicalai.data.gym import GymDataset
from physicalai.data.observation import Observation
from physicalai.gyms.step_limit import with_step_limit

if TYPE_CHECKING:
    from physicalai.gyms import Gym

logger = logging.getLogger(__name__)

#: Maximum number of workers used by the ``"auto"`` heuristic.
_AUTO_NUM_WORKERS_CAP = 8


def resolve_auto_num_workers() -> int:
    """Resolve the number of DataLoader workers for ``"auto"`` mode.

    Returns:
        int: Number of workers to use.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 0
    return min(cpu_count, _AUTO_NUM_WORKERS_CAP)


def _collate_gym(batch: list[Any]) -> Gym:
    """Collate a batch of environments into a single Gym environment.

    Args:
        batch: A list containing a single Gym environment.

    Returns:
        Gym: The gym environment (unwrapped from batch list).
    """
    # batch is a list with one item: [env], return it directly
    return batch[0]


def _collate_observations(batch: list[Observation]) -> Observation:
    """Collate a batch of Observations into a single batched Observation.

    Args:
        batch (list[Observation]): A list containing Observations.

    Returns:
        Observation: A single Observation with batched tensors.
    """
    if not batch:
        return Observation()

    collated_data: dict[str, Any] = {}

    # Iterate through all fields defined in the Observation dataclass
    for field in fields(Observation):
        key = field.name
        values = [getattr(elem, key) for elem in batch]

        # Filter out None values to determine the data type
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            collated_data[key] = None
            continue

        first_non_none = non_none_values[0]

        # Handle tensors and NumPy arrays
        if isinstance(first_non_none, (torch.Tensor, np.ndarray)):
            # Convert NumPy arrays to PyTorch tensors before stacking
            tensors_to_stack = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in non_none_values]
            collated_data[key] = torch.stack(tensors_to_stack, dim=0)

        # Handle nested dictionaries, such as the `images` field
        elif isinstance(first_non_none, dict):
            collated_inner_dict: dict[str, Any] = {}
            for inner_key in first_non_none:
                inner_values = [d.get(inner_key) for d in values if d is not None]
                if inner_values:
                    first_inner_value = inner_values[0]
                    # Only stack if the values are tensors or arrays
                    if isinstance(first_inner_value, (torch.Tensor, np.ndarray)):
                        tensors_to_stack = [
                            torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in inner_values
                        ]
                        collated_inner_dict[inner_key] = torch.stack(tensors_to_stack, dim=0)
                    else:
                        # For non-tensor values (like strings), just keep them as a list
                        collated_inner_dict[inner_key] = inner_values
            collated_data[key] = collated_inner_dict

        # Handle primitive types like booleans, integers, and floats
        elif isinstance(first_non_none, (bool, int, float)):
            collated_data[key] = torch.tensor(non_none_values)

        # Fallback for other types, like strings
        else:
            collated_data[key] = values

    return Observation(**collated_data)


class DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for action datasets and Gym environments.

    Handles training, evaluation, and test datasets, including Gym environments
    wrapped as datasets. Provides DataLoaders for training, validation, and testing.
    """

    def __init__(
        self,
        train_dataset: TorchDataset[Any],
        train_batch_size: int = 16,
        num_workers: int | Literal["auto"] = "auto",
        val_gym: Gym | None = None,
        num_rollouts_val: int = 10,
        test_gym: Gym | None = None,
        num_rollouts_test: int = 10,
        max_episode_steps: int | None = 300,
    ) -> None:
        """Initialize the ActionDataModule.

        Args:
            train_dataset (ActionDataset): Dataset for training.
            train_batch_size (int): Batch size for training DataLoader.
            num_workers (int | Literal["auto"]): Number of DataLoader workers.
                ``"auto"`` (default) uses ``min(cpu_count, 8)``.
            val_gym (Gym | None): Validation environment.
            num_rollouts_val (int): Number of rollouts to run for validation environments.
            test_gym (Gym | None): Test environment.
            num_rollouts_test (int): Number of rollouts to run for test environments.
            max_episode_steps (int, None): Maximum steps allowed per episode. If None, no time limit.
        """
        super().__init__()

        # dataset
        self.train_dataset: TorchDataset[Any] = train_dataset
        self.train_batch_size: int = train_batch_size
        self.num_workers: int = resolve_auto_num_workers() if num_workers == "auto" else num_workers
        logger.info("DataLoader workers: %d%s", self.num_workers, " (auto)" if num_workers == "auto" else "")

        # gym environments
        self.val_gym: Gym | None = val_gym
        self.val_dataset: TorchDataset[Any] | None = None
        self.num_rollouts_val: int = num_rollouts_val
        self.test_gym: Gym | None = test_gym
        self.test_dataset: TorchDataset[Any] | None = None
        self.num_rollouts_test: int = num_rollouts_test
        self.max_episode_steps = max_episode_steps

        # setup time limit if max_episode steps (0 or None will fail)
        if self.max_episode_steps:
            self.val_gym = with_step_limit(self.val_gym, max_steps=self.max_episode_steps) if self.val_gym else None
            self.test_gym = with_step_limit(self.test_gym, max_steps=self.max_episode_steps) if self.test_gym else None

    @property
    def batch_size(self) -> int:
        """Alias for ``train_batch_size``.

        Lightning's ``Tuner.scale_batch_size`` reads and writes this attribute.
        """
        return self.train_batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.train_batch_size = value

    def setup(self, stage: str) -> None:
        """Set up datasets depending on the stage (fit or test).

        Args:
            stage (str): Stage of training ('fit', 'test', etc.).
        """
        if stage == "fit" and self.val_gym:
            self.val_dataset = GymDataset(
                env=self.val_gym,
                num_rollouts=self.num_rollouts_val,
            )

        if stage == "test" and self.test_gym:
            self.test_dataset = GymDataset(
                env=self.test_gym,
                num_rollouts=self.num_rollouts_test,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for training.

        Returns:
            DataLoader[Any]: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_observations,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for validation.

        If no validation dataset is configured, returns an empty DataLoader
        to allow training to proceed without validation.

        Returns:
            DataLoader[Any]: Validation DataLoader with collate function for Gym environments,
                           or empty DataLoader if no validation dataset is configured.
        """
        if self.val_dataset is None:
            # Return empty dataloader when no validation dataset
            # This allows training to proceed without validation
            return DataLoader(TensorDataset(torch.empty(0)), batch_size=1)

        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=_collate_gym,  # type: ignore[arg-type]
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for testing.

        If no test dataset is configured, returns an empty DataLoader
        to allow training to proceed without testing.

        Returns:
            DataLoader[Any]: Test DataLoader with collate function for Gym environments,
                           or empty DataLoader if no test dataset is configured.
        """
        if self.test_dataset is None:
            # Return empty dataloader when no test dataset
            # This allows training to proceed without testing
            return DataLoader(TensorDataset(torch.empty(0)), batch_size=1)

        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=_collate_gym,  # type: ignore[arg-type]
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Predict DataLoader is not implemented.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
