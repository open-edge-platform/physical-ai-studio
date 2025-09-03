# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""
Lightning datamodules:
    LerobotDataset/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from action_trainer.data.gym import GymDataset

if TYPE_CHECKING:
    from action_trainer.data import ActionDataset
    from action_trainer.gyms import BaseGym


def collate_env(batch: list[Any]) -> dict[str, Any]:
    """
    Collate a batch of environments for a DataLoader.

    Args:
        batch (list[Any]): A list containing a single environment.

    Returns:
        dict[str, Any]: Dictionary with the environment under the key 'env'.
    """
    # batch is a list with one item: [env], return a dict as expected by test_step
    return {"env": batch[0]}


class LeRobotDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for LeRobot datasets and Gym environments.

    Handles training, evaluation, and test datasets, including Gym environments
    wrapped as datasets. Provides DataLoaders for training, validation, and testing.
    """

    def __init__(
        self,
        train_dataset: ActionDataset,
        train_batch_size: int,
        eval_gyms: BaseGym | list[BaseGym] | None = None,
        num_rollouts_eval: int = 10,
        test_gyms: BaseGym | list[BaseGym] | None = None,
        num_rollouts_test: int = 10,
    ) -> None:
        """
        Initialize the LeRobotDataModule.

        Args:
            train_dataset (LeRobotDataset): Dataset for training.
            train_batch_size (int): Batch size for training DataLoader.
            eval_gyms (BaseGym, list[BaseGym], None]): Evaluation environments.
            test_gyms (BaseGym, list[BaseGym], None]): Test environments.
        """
        super().__init__()

        # dataset
        self.train_dataset: ActionDataset = train_dataset
        self.train_batch_size: int = train_batch_size

        # gym environments
        self.eval_gyms: BaseGym | list[BaseGym] | None = eval_gyms
        self.eval_dataset: Optional[Dataset[Any]] = None
        self.num_rollouts_eval: int = num_rollouts_eval
        self.test_gyms: BaseGym | list[BaseGym] | None = test_gyms
        self.test_dataset: Optional[Dataset[Any]] = None
        self.num_rollouts_test: int = num_rollouts_test

        self.save_hyperparameters(ignore=["eval_gyms", "test_gyms"])

    def setup(self, stage: str) -> None:
        """
        Set up datasets depending on the stage (fit or test).

        Args:
            stage (str): Stage of training ('fit', 'test', etc.).
        """
        if stage == "fit" and self.eval_gyms:
            if isinstance(self.eval_gyms, list):
                self.eval_dataset = ConcatDataset([
                    GymDataset(env=gym, num_rollouts=self.num_rollouts_eval) for gym in self.eval_gyms
                ])
            else:
                self.eval_dataset = GymDataset(env=self.eval_gyms, num_rollouts=self.num_rollouts_eval)

        if stage == "test" and self.test_gyms:
            if isinstance(self.test_gyms, list):
                self.test_dataset = ConcatDataset([
                    GymDataset(env=gym, num_rollouts=self.num_rollouts_test) for gym in self.test_gyms
                ])
            else:
                self.test_dataset = GymDataset(env=self.test_gyms, num_rollouts=self.num_rollouts_test)

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Return the DataLoader for training.

        Returns:
            DataLoader[Any]: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            num_workers=4,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Return the DataLoader for validation.

        Returns:
            DataLoader[Any]: Validation DataLoader with collate function for Gym environments.
        """
        return DataLoader(
            self.eval_dataset,
            batch_size=1,
            collate_fn=collate_env,  # type: ignore[arg-type]
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Return the DataLoader for testing.

        Returns:
            DataLoader[Any]: Test DataLoader with collate function for Gym environments.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=collate_env,  # type: ignore[arg-type]
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """
        Predict DataLoader is not implemented.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
