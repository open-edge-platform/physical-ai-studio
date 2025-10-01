# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Lightning datamodules."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from getiaction.data import Observation
from getiaction.data.gym import GymDataset
from getiaction.gyms import BaseGym

if TYPE_CHECKING:
    from getiaction.data import Dataset


def _collate_env(batch: list[Any]) -> dict[str, Any]:
    """Collate a batch of environments for a DataLoader.

    Args:
        batch (list[Any]): A list containing a single environment.

    Returns:
        dict[str, Any]: Dictionary with the environment under the key 'env'.
    """
    # batch is a list with one item: [env], return a dict as expected by test_step
    return {"env": batch[0]}


def _collate_observations(batch: list[Observation]) -> dict[str, Any]:
    """Collate a batch of Observations to a dict for training format.

    Args:
        batch (list[Any]): A list containing Observations.

    Returns:
        dict[str, Any]: Dictionary for use in the model.
    """
    if not batch:
        return {}

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

    return collated_data


class DataModule(LightningDataModule):
    """PyTorch Lightning DataModule for action datasets and Gym environments.

    Handles training, evaluation, and test datasets, including Gym environments
    wrapped as datasets. Provides DataLoaders for training, validation, and testing.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        train_batch_size: int = 16,
        eval_gyms: BaseGym | list[BaseGym] | None = None,
        num_rollouts_eval: int = 10,
        test_gyms: BaseGym | list[BaseGym] | None = None,
        num_rollouts_test: int = 10,
        max_episode_steps: int | None = 300,
    ) -> None:
        """Initialize the ActionDataModule.

        Args:
            train_dataset (ActionDataset): Dataset for training.
            train_batch_size (int): Batch size for training DataLoader.
            eval_gyms (BaseGym, list[BaseGym], None]): Evaluation environments.
            num_rollouts_eval (int): Number of rollouts to run for evaluation environments.
            test_gyms (BaseGym, list[BaseGym], None]): Test environments.
            num_rollouts_test (int): Number of rollouts to run for test environments.
            max_episode_steps (int, None): Maximum steps allowed per episode. If None, no time limit.
        """
        super().__init__()

        # dataset
        self.train_dataset: Dataset = train_dataset
        self.train_batch_size: int = train_batch_size

        # gym environments
        self.eval_gyms: BaseGym | list[BaseGym] | None = eval_gyms
        self.eval_dataset: Dataset[Any] | None = None
        self.num_rollouts_eval: int = num_rollouts_eval
        self.test_gyms: BaseGym | list[BaseGym] | None = test_gyms
        self.test_dataset: Dataset[Any] | None = None
        self.num_rollouts_test: int = num_rollouts_test
        self.max_episode_steps = max_episode_steps

        # setup time limit if max_episode steps
        if (self.max_episode_steps is not None) and self.eval_gyms is not None:
            if isinstance(self.eval_gyms, BaseGym):
                self.eval_gyms.env = TimeLimit(
                    env=self.eval_gyms.env,
                    max_episode_steps=self.max_episode_steps,
                )
            elif isinstance(self.eval_gyms, list):
                for eval_gym in self.eval_gyms:
                    eval_gym.env = TimeLimit(
                        env=eval_gym.env,
                        max_episode_steps=self.max_episode_steps,
                    )
        if (self.max_episode_steps is not None) and self.test_gyms is not None:
            if isinstance(self.test_gyms, BaseGym):
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
        if stage == "fit" and self.eval_gyms:
            if isinstance(self.eval_gyms, list):
                # TODO(alfie-roddan-intel): https://github.com/open-edge-platform/geti-action/issues/33  # noqa: FIX002
                # ensure metrics are seperable between two different gyms
                self.eval_dataset = ConcatDataset([
                    GymDataset(env=gym, num_rollouts=self.num_rollouts_eval) for gym in self.eval_gyms
                ])
            else:
                self.eval_dataset = GymDataset(env=self.eval_gyms, num_rollouts=self.num_rollouts_eval)

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
            self.eval_dataset,
            batch_size=1,
            collate_fn=_collate_env,  # type: ignore[arg-type]
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
            collate_fn=_collate_env,  # type: ignore[arg-type]
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Predict DataLoader is not implemented.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
