# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Lightning backend"""

import lightning as L

from action_trainer.data import DataModule
from action_trainer.policies.base import TrainerModule
from action_trainer.train.callbacks import PolicyDatasetInteraction


class Trainer:
    """Lightning Trainer wrapper with policy-datamodule interaction hook."""

    def __init__(
        self,
        num_sanity_val_steps: int = 0,
        callbacks: list | bool | None = None,
        **trainer_kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the Trainer.

        Args:
            num_sanity_val_steps (int): Number of validation sanity steps.
            callbacks (list, optional): User callbacks. Defaults to None.
            **trainer_kwargs: Other Lightning Trainer kwargs.
        """
        if callbacks is None:
            callbacks = []
            callbacks.append(PolicyDatasetInteraction())
        elif isinstance(callbacks, list):
            callbacks.append(PolicyDatasetInteraction())

        self.backend = L.Trainer(
            callbacks=callbacks,
            num_sanity_val_steps=num_sanity_val_steps,
            **trainer_kwargs,
        )

    def fit(self, model: TrainerModule, datamodule: DataModule, **kwargs) -> None:  # noqa: ANN003
        """Fit the model."""
        # if we don't have any validation datasets, limit batch size to zero
        if datamodule.eval_dataset is None:
            self.backend.limit_val_batches = 0
        return self.backend.fit(model=model, datamodule=datamodule, **kwargs)

    def test(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Test the model."""
        del args, kwargs
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Predict the model."""
        del args, kwargs
        raise NotImplementedError

    def validate(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Validate the model."""
        del args, kwargs
        raise NotImplementedError
