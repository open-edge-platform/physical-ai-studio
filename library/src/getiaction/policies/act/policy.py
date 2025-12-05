# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning module for ACT policy."""

from pathlib import Path
from typing import Any, Self

import torch
import yaml

from getiaction.data import Dataset, Observation
from getiaction.export import Export, FromCheckpoint
from getiaction.export.mixin_export import CONFIG_KEY as _MODEL_CONFIG_KEY
from getiaction.gyms import Gym
from getiaction.policies.act.config import ACTConfig
from getiaction.policies.act.model import ACT as ACTModel  # noqa: N811
from getiaction.policies.base import Policy
from getiaction.train.utils import reformat_dataset_to_match_policy


class ACT(FromCheckpoint, Export, Policy):  # type: ignore[misc]
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

        Export examples:
        >>> policy = ACT(model)
        >>> # Export to OpenVINO (recommended for Intel platforms)
        >>> policy.export("./exports", backend="openvino")
        >>> # Export to ONNX (cross-platform)
        >>> policy.export("./exports", backend="onnx")
    """

    model_type: type = ACTModel
    model_config_type: type = ACTConfig

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

        self.model = model  # type: ignore[assignment]
        self.optimizer = optimizer

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        config: dict[str, Any] | str | Path | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Self:
        """Create ACT policy from a dataset with eager initialization.

        This factory method extracts features from the dataset and builds the model
        immediately, making the policy ready for inference without a Trainer.

        Args:
            dataset: Dataset to extract features from. Must implement
                `observation_features` and `action_features` properties.
            config: Optional config dict or path to YAML file for ACTModel.
                If None, uses default model parameters.
            **kwargs: Additional keyword arguments to override config parameters.

        Returns:
            Fully initialized ACT policy ready for inference.

        Examples:
            Create policy with default parameters:

                >>> from getiaction.policies import ACT
                >>> policy = ACT.from_dataset(dataset)
                >>> action = policy.select_action(observation)

            Create policy with custom model parameters:

                >>> policy = ACT.from_dataset(dataset, dim_model=256, chunk_size=50)

            Create policy from YAML config file:

                >>> policy = ACT.from_dataset(dataset, config="configs/act.yaml")

            Create policy with config and overrides:

                >>> policy = ACT.from_dataset(
                ...     dataset,
                ...     config="configs/act.yaml",
                ...     chunk_size=100,  # Override config value
                ... )
        """
        input_features = dataset.observation_features
        output_features = dataset.action_features

        # Load config if path provided
        model_kwargs: dict[str, Any] = {}
        if config is not None:
            if isinstance(config, (str, Path)):
                with Path(config).open("r", encoding="utf-8") as f:
                    model_kwargs = yaml.safe_load(f)
            else:
                model_kwargs = dict(config)

        # Merge with kwargs (kwargs override config)
        model_kwargs.update(kwargs)

        # Build the model
        model = ACTModel(
            input_features=input_features,
            output_features=output_features,
            **model_kwargs,
        )

        return cls(model=model)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save model config to checkpoint for reconstruction.

        Converts ACTConfig dataclass to a plain dict for safe serialization.
        This allows torch.load() to use weights_only=True for security.
        """
        if self.model is not None and hasattr(self.model, "config"):
            checkpoint[_MODEL_CONFIG_KEY] = self.model.config.to_dict()

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

        datamodule = self.trainer.datamodule  # type: ignore[union-attr]
        train_dataset = datamodule.train_dataset

        # Get the underlying LeRobot dataset - handle both data formats
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected train_dataset to be getiaction.data.Dataset, got {type(train_dataset)}."
            raise TypeError(msg)

        # Initialize the policy
        self.model = ACTModel(
            input_features=train_dataset.observation_features,
            output_features=train_dataset.action_features,
        )

        # TO-DO(Vlad):  remove that workaround after CLI is able to run getiaction trainer
        reformat_dataset_to_match_policy(self, datamodule)

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch: Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        # Move batch to device (observations from gym are on CPU)
        batch = batch.to(self.device)
        chunk = self.model.predict_action_chunk(batch.to_dict())

        # select only first action from the predicted chunk to unify output with other policies
        return chunk[:, 1, :]

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Perform forward pass of the ACT policy.

        The return value depends on the model's training mode:
        - In training mode: Returns (loss, loss_dict) from the model's forward method
        - In evaluation mode: Returns action predictions via select_action method

        Args:
            batch (Observation): Input batch of observations

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: In training mode, returns
                tuple of (loss, loss_dict). In eval mode, returns selected actions tensor.
        """
        if self.training:
            # During training, return loss information for backpropagation
            return self.model(batch.to_dict())

        # During evaluation, return action predictions
        return self.select_action(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Observation): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        del batch_idx
        loss, loss_dict = self.model(batch.to_dict())  # noqa: RUF059
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

    def evaluation_step(self, batch: Observation, stage: str) -> None:  # noqa: PLR6301
        """Evaluation step (no-op by default).

        Args:
            batch (Observation): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step.

        Runs gym-based validation via rollout evaluation. The DataModule's val_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step.

        Runs gym-based testing via rollout evaluation. The DataModule's test_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def reset(self) -> None:
        """Reset the policy state for a new episode.

        Clears internal state like action queues or observation history.
        For ACT, this delegates to the model's reset method.
        """
        if hasattr(self.model, "reset") and callable(self.model.reset):
            self.model.reset()
