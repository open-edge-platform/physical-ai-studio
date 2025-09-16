# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Lightning backend"""

from typing import Callable

import lightning as L
from lightning.pytorch.callbacks import Callback

from action_trainer.data import ActionDataModule
from action_trainer.policies.base import ActionTrainerModule
from action_trainer.train.utils import reformat_dataset_to_match_policy


class PolicyDatasetInteractionCallback(Callback):
    """Callback to interact the policy and dataset before training starts."""

    def __init__(self, hook_fn: Callable[[L.Trainer, L.LightningModule], None]):
        """
        Args:
            hook_fn (Callable): Function that takes (trainer, model) and
                performs the interaction, e.g., calling
                `reformat_dataset_to_match_policy`.
        """
        self.hook_fn = hook_fn

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self.hook_fn(trainer, pl_module)


class LightningActionTrainer:
    """Lightning Trainer wrapper with policy-datamodule interaction hook."""

    def __init__(
        self,
        num_sanity_val_steps: int = 0,
        callbacks: list | None = None,
        **trainer_kwargs,
    ):
        """
        Args:
            num_sanity_val_steps (int): Number of validation sanity steps.
            callbacks (list, optional): User callbacks. Defaults to None.
            **trainer_kwargs: Other Lightning Trainer kwargs.
        """
        if callbacks is None:
            callbacks = []

        # Add the policy-dataset interaction callback automatically
        def _interact_policy_dataset(trainer: L.Trainer, model: L.LightningModule):
            # Assumes trainer has a datamodule attached
            if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

        callbacks.append(PolicyDatasetInteractionCallback(_interact_policy_dataset))

        self.trainer = L.Trainer(
            callbacks=callbacks,
            num_sanity_val_steps=num_sanity_val_steps,
            **trainer_kwargs,
        )

    def fit(self, model: ActionTrainerModule, datamodule: ActionDataModule, **kwargs):
        # if we don't have any validation datasets, limit batch size to zero
        if datamodule.eval_dataset is None:
            self.trainer.limit_val_batches = 0
        return self.trainer.fit(model=model, datamodule=datamodule, **kwargs)

    def test(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError

    def validate(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError
