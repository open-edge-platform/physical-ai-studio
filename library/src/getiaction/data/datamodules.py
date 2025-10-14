# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Lightning datamodules."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from getiaction.data.gym import GymDataset
from getiaction.data.observation import Observation
from getiaction.gyms import Gym

if TYPE_CHECKING:
    from getiaction.data import Dataset


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
            collated_inner_dict = {}
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
        train_dataset: Dataset,
        train_batch_size: int = 16,
        val_gyms: Gym | list[Gym] | None = None,
        num_rollouts_val: int = 10,
        test_gyms: Gym | list[Gym] | None = None,
        num_rollouts_test: int = 10,
        max_episode_steps: int | None = 300,
    ) -> None:
        """Initialize the ActionDataModule.

        Args:
            train_dataset (ActionDataset): Dataset for training.
            train_batch_size (int): Batch size for training DataLoader.
            val_gyms (Gym, list[Gym], None]): Validation environments.
            num_rollouts_val (int): Number of rollouts to run for validation environments.
            test_gyms (Gym, list[Gym], None]): Test environments.
            num_rollouts_test (int): Number of rollouts to run for test environments.
            max_episode_steps (int, None): Maximum steps allowed per episode. If None, no time limit.
        """
        super().__init__()

        # dataset
        self.train_dataset: Dataset = train_dataset
        self.train_batch_size: int = train_batch_size

        # gym environments
        self.val_gyms: Gym | list[Gym] | None = val_gyms
        self.val_dataset: Dataset | None = None
        self.num_rollouts_val: int = num_rollouts_val
        self.test_gyms: Gym | list[Gym] | None = test_gyms
        self.test_dataset: Dataset | None = None
        self.num_rollouts_test: int = num_rollouts_test
        self.max_episode_steps = max_episode_steps

        # setup time limit if max_episode steps
        if (self.max_episode_steps is not None) and self.val_gyms is not None:
            if isinstance(self.val_gyms, Gym):
                self.val_gyms.env = TimeLimit(
                    env=self.val_gyms.env,
                    max_episode_steps=self.max_episode_steps,
                )
            elif isinstance(self.val_gyms, list):
                for val_gym in self.val_gyms:
                    val_gym.env = TimeLimit(
                        env=val_gym.env,
                        max_episode_steps=self.max_episode_steps,
                    )
        if (self.max_episode_steps is not None) and self.test_gyms is not None:
            if isinstance(self.test_gyms, Gym):
                self.test_gyms.env = TimeLimit(
                    env=self.test_gyms.env,
                    max_episode_steps=self.max_episode_steps,
                )
            elif isinstance(self.test_gyms, list):
                for test_gym in self.test_gyms:
                    test_gym.env = TimeLimit(
                        env=test_gym.env,
                        max_episode_steps=self.max_episode_steps,
                    )

    def setup(self, stage: str) -> None:
        """Set up datasets depending on the stage (fit or test).

        Args:
            stage (str): Stage of training ('fit', 'test', etc.).
        """
        if stage == "fit" and self.val_gyms:
            if isinstance(self.val_gyms, list):
                # TODO(alfie-roddan-intel): https://github.com/open-edge-platform/geti-action/issues/33  # noqa: FIX002
                # ensure metrics are seperable between two different gyms
                self.val_dataset = ConcatDataset([
                    GymDataset(env=gym, num_rollouts=self.num_rollouts_val) for gym in self.val_gyms
                ])
            else:
                self.val_dataset = GymDataset(env=self.val_gyms, num_rollouts=self.num_rollouts_val)

        if stage == "test" and self.test_gyms:
            if isinstance(self.test_gyms, list):
                # TODO(alfie-roddan-intel): https://github.com/open-edge-platform/geti-action/issues/33  # noqa: FIX002
                # ensure metrics are seperable between two different gyms
                self.test_dataset = ConcatDataset([
                    GymDataset(env=gym, num_rollouts=self.num_rollouts_test) for gym in self.test_gyms
                ])
            else:
                self.test_dataset = GymDataset(env=self.test_gyms, num_rollouts=self.num_rollouts_test)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for training.

        Returns:
            DataLoader[Any]: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            num_workers=4,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_observations,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for validation.

        Returns:
            DataLoader[Any]: Validation DataLoader with collate function for Gym environments.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=_collate_gym,  # type: ignore[arg-type]
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the DataLoader for testing.

        Returns:
            DataLoader[Any]: Test DataLoader with collate function for Gym environments.
        """
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
