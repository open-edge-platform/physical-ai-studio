# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Callbacks for training
"""

import lightning as L
from lightning.pytorch.callbacks import Callback

from action_trainer.train.utils import reformat_dataset_to_match_policy


class PolicyDatasetInteraction(Callback):
    """Callback to interact the policy and dataset before training starts."""

    def _interact_policy_dataset(self, trainer: L.Trainer, model: L.LightningModule):
        # Assumes trainer has a datamodule attached
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self._interact_policy_dataset(trainer, pl_module)
