# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Lightning backend"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from action_trainer.data.datamodules import collate_env
from action_trainer.data.gym import GymDataset
from action_trainer.train.callbacks import ActionMetricsCallback

if TYPE_CHECKING:
    from action_trainer.data import ActionDataModule
    from action_trainer.gyms import BaseGym
    from action_trainer.policies.base.base_lightning_module import ActionTrainerModule


class LightningActionTrainer:
    _NO_DEFAULT = object()  # sentinel to detect "not passed"

    def __init__(
        self,
        datamodule: ActionDataModule,
        model: ActionTrainerModule,
        num_sanity_val_steps: int = 0,
        callbacks: Any = _NO_DEFAULT,  # if user sets callbacks=None we will not override
        **lightning_trainer_kwargs,
    ):
        if callbacks is self._NO_DEFAULT:
            callbacks = [
                ActionMetricsCallback("val"),
                ActionMetricsCallback("test"),
                LearningRateMonitor("step"),
            ]
        self.datamodule = datamodule
        self.model = model
        self.model, self.datamodule = self.reformat_dataset_to_match_policy(
            model=self.model,
            datamodule=self.datamodule,
        )
        self.trainer = lightning.Trainer(
            callbacks=callbacks,
            num_sanity_val_steps=num_sanity_val_steps,
            **lightning_trainer_kwargs,
        )

    @staticmethod
    def reformat_dataset_to_match_policy(model: ActionTrainerModule, datamodule: ActionDataModule):
        # do something here
        return model, datamodule

    def fit(self, **lightning_fit_kwargs):
        self.trainer.fit(
            datamodule=self.datamodule,
            model=self.model,
            **lightning_fit_kwargs,
        )

    def test(self, env: BaseGym | None = None, num_rollouts: int | None = None, **lightning_test_kwargs):
        if (env is not None) and (num_rollouts is not None):
            test_dataset = GymDataset(env=env, num_rollouts=num_rollouts)
            test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_env, shuffle=False)
            self.trainer.test(model=self.model, dataloaders=test_dataloader, **lightning_test_kwargs)
        else:
            self.trainer.test(model=self.model, datamodule=self.datamodule, **lightning_test_kwargs)

    def predict(self):
        raise NotImplementedError
