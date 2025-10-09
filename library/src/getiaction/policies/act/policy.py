# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for ACT policy."""

import torch

from getiaction.data import Dataset
from getiaction.policies.act.model import ACT as ACTModel  # noqa: N811
from getiaction.policies.base import Policy
from getiaction.train.utils import reformat_dataset_to_match_policy


class ACT(Policy):
    """Action Chunking with Transformers (ACT) policy implementation.

    This class implements the ACT policy for imitation learning, which uses a transformer-based
    architecture to predict sequences of actions given observations.
    Policy contains contains model and other related modules and methods that are required
    to start training in a Lightning Trainer.

    Example:
        >>> model = ACTModel(...)
        >>> policy = ACT(model)
        >>> actions = policy.select_action(batch)
        >>> loss_dict = policy.training_step(batch, batch_idx=0)
    """

    def __init__(
        self,
        model: ACTModel | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize the ACT policy with a model and optional optimizer.

        Args:
            model (ACTModel): The ACT model to be used by this policy.
            optimizer (torch.optim.Optimizer | None, optional): The optimizer for training.
                If None, defaults to Adam optimizer with lr=1e-5 and weight_decay=1e-4.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer

    def setup(self, stage: str) -> None:
        """Set up the policy from datamodule if not already initialized.

        This method is called by Lightning before fit/validate/test/predict.
        It extracts features from the datamodule's training dataset and
        initializes the policy if it wasn't already created in __init__.

        Args:
            stage: The stage of training ('fit', 'validate', 'test', or 'predict')

        Raises:
            TypeError: If the train_dataset is not a getiaction.data.Dataset.
        """
        del stage  # Unused argument

        if self.model is not None:
            return  # Already initialized

        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both data formats
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected train_dataset to be getiaction.data.Dataset, got {type(train_dataset)}."
            raise TypeError(msg)

        # Initialize the policy
        self.model = ACTModel(
            action_features=train_dataset.action_features,
            observation_features=train_dataset.observation_features,
        )

        # Workaround to run via lightning CLI
        reformat_dataset_to_match_policy(self, datamodule)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (dict[str, torch.Tensor]): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        return self.model.predict_action_chunk(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (dict[str, torch.Tensor]): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        del batch_idx
        loss, loss_dict = self.forward(batch)  # noqa: RUF059
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the policy.

        Returns:
            torch.optim.Optimizer: Adam optimizer over the model parameters.
        """
        if self.optimizer is not None:
            return self.optimizer
        return torch.optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-4)

    def evaluation_step(self, batch: dict[str, torch.Tensor], stage: str) -> None:  # noqa: PLR6301
        """Evaluation step (no-op by default).

        Args:
            batch (dict[str, torch.Tensor]): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step (calls evaluation_step).

        Args:
            batch (dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        del batch_idx
        return self.evaluation_step(batch=batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step (calls evaluation_step).

        Args:
            batch (dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        del batch_idx
        return self.evaluation_step(batch=batch, stage="test")
