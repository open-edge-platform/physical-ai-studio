# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage."""

from collections.abc import Iterable

import torch

from getiaction.data import Observation
from getiaction.export.mixin_export import Export
from getiaction.gyms import Gym
from getiaction.policies.base import Policy
from getiaction.policies.dummy.config import DummyConfig
from getiaction.policies.dummy.model import Dummy as DummyModel


class Dummy(Export, Policy):
    """Dummy policy wrapper."""

    def __init__(self, config: DummyConfig) -> None:
        """Initialize the Dummy policy wrapper.

        This class wraps a `DummyModel` and integrates it into a `TrainerModule`,
        validating the action shape and preparing the model for training.

        Args:
            config (DummyConfig): Configuration object containing the action shape
                and other hyperparameters required for initializing the policy.
        """
        super().__init__()
        self.config = config
        self.action_shape = self._validate_action_shape(self.config.action_shape)
        self.action_dtype = self._validate_dtype(self.config.action_dtype)
        self.action_min, self.action_max = self._validate_min_max(
            min_=self.config.action_min,
            max_=self.config.action_max,
        )

        # model
        self.model = DummyModel(
            action_shape=self.action_shape,
            action_dtype=self.action_dtype,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    @staticmethod
    def _validate_action_shape(shape: list | tuple) -> list | tuple:
        """Validate and normalize the action shape.

        Args:
            shape (list | tuple): The input shape to validate.

        Returns:
            list | tuple: A validated list or tuple object.

        Raises:
            ValueError: If `shape` is `None`.
            TypeError: If `shape` is not a valid type (e.g., string).
        """
        if shape is None:
            msg = "Action is missing a 'shape' key in its features dictionary."
            raise ValueError(msg)

        if isinstance(shape, torch.Size):
            return shape

        if isinstance(shape, str):
            msg = f"Shape for action '{shape}' must be a sequence of numbers, but received a string."
            raise TypeError(msg)

        if isinstance(shape, Iterable):
            return list(shape)

        msg = f"The 'action_shape' argument must be a list or tuple, but received type {type(shape).__name__}."
        raise TypeError(msg)

    @staticmethod
    def _validate_dtype(dtype: torch.dtype | str | None) -> torch.dtype:
        """Validate and resolve dtype.

        Args:
            dtype: The dtype to validate. May be a ``torch.dtype`` instance,
                a string representing a dtype (e.g., ``"float32"``,
                ``"double"``, ``"int"``), or ``None``.

        Returns:
            torch.dtype: A fully-resolved PyTorch dtype.

        Raises:
            ValueError: If the provided string cannot be resolved to a valid
                ``torch.dtype``.
        """
        # if None, assume float32
        if dtype is None:
            return torch.float32

        # if already a dtype then return
        if isinstance(dtype, torch.dtype):
            return dtype

        # ensure string and lower
        key = str(dtype).lower()

        # common aliases for dtypes
        alias_map = {
            "float": "float32",
            "fp32": "float32",
            "double": "float64",
            "fp64": "float64",
            "half": "float16",
            "long": "int64",
            "int": "int32",
            "short": "int16",
            "byte": "uint8",
            "bf16": "bfloat16",
        }
        key = alias_map.get(key, key)

        attr = getattr(torch, key, None)
        if isinstance(attr, torch.dtype):
            return attr

        msg = f"Unknown dtype string: {dtype}"
        raise ValueError(msg)

    @staticmethod
    def _validate_min_max(
        min_: float | None = None,
        max_: float | None = None,
    ) -> tuple[float | None, float | None]:
        """Validate range for action space.

        Args:
            min_: The lower bound of the range, or ``None``.
            max_: The upper bound of the range, or ``None``.

        Returns:
            tuple[float | None, float | None]: A tuple ``(min_, max_)`` where
                both values are either the validated inputs or ``(None, None)``
                if the range is unspecified.

        Raises:
            ValueError: If both bounds are provided and ``max_`` is smaller
                than ``min_``.
        """
        if (min_ is None) or (max_ is None):
            return (min_, max_)
        # only assumption is that min is smaller than max
        if max_ < min_:
            msg = f"Max cannot be smaller than min: {max_} < {min_}"
            raise ValueError(msg)
        return (min_, max_)

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch: Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        # Get action from model
        action = self.model.select_action(batch.to_dict())  # type: ignore[attr-defined]

        # Remove batch dim if present (rollout expects unbatched)
        return action.squeeze(0) if action.ndim > 1 and action.shape[0] == 1 else action

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Perform forward pass of the Dummy policy.

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

    @staticmethod
    def _generate_example_inputs() -> dict[str, torch.Tensor]:
        """Generate example inputs for export.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with dummy observation inputs.
        """
        return {"state": torch.randn(1, 4)}

    def training_step(self, batch: Observation, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Observation): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        del batch_idx
        loss, _ = self.model(batch.to_dict())
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
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    @staticmethod
    def evaluation_step(batch: Observation, stage: str) -> None:
        """Evaluation step (no-op by default).

        Args:
            batch (Observation): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage  # Unused variables

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
        """Reset the policy state.

        Dummy policy has no state to reset, so this is a no-op.
        """
