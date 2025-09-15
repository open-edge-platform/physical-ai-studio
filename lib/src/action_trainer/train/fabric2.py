# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer with Fabric backend"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning import Fabric
from torch.utils.data import DataLoader
from tqdm import tqdm

from action_trainer.data.datamodules import collate_env
from action_trainer.data.gym import GymDataset

if TYPE_CHECKING:
    from action_trainer.data import ActionDataModule
    from action_trainer.policies.base.base_lightning_module import ActionTrainerModule


class FabricActionTrainer:
    _NO_DEFAULT = object()

    def __init__(
        self,
        datamodule: ActionDataModule,
        model: ActionTrainerModule,
        max_epochs: int = 2,
        check_val_every_n_epoch: int = 1,
        **fabric_kwargs,
    ):
        self.datamodule = datamodule
        self.model = model
        self.model, self.datamodule = self.reformat_dataset_to_match_policy(model, datamodule)

        # Fabric setup
        self.fabric = Fabric(**fabric_kwargs)

        # Optimizer
        self.optimizer = self.model.configure_optimizers()

        # Setup model + optimizer on devices
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Datamodule setup for train/val
        if hasattr(self.datamodule, "setup"):
            self.datamodule.setup(stage="fit")

        self.train_dataloader = self.fabric.setup_dataloaders(self.datamodule.train_dataloader())
        self.val_dataloader = self.fabric.setup_dataloaders(self.datamodule.val_dataloader())
        self.test_dataloader = None  # optional

        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.global_step = 0

    @staticmethod
    def reformat_dataset_to_match_policy(model, datamodule):
        return model, datamodule

    def fit(self):
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loader = tqdm(self.train_dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                step_out = self.model.training_step(batch, batch_idx)
                loss = step_out["loss"]

                self.optimizer.zero_grad()
                self.fabric.backward(loss)
                self.optimizer.step()

                # Log all metrics in the step dict
                self.fabric.log_dict(
                    {f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_out.items()},
                    step=self.global_step,
                )

                epoch_loss += loss.item()
                n_batches += 1
                self.global_step += 1
                train_loader.set_postfix(loss=epoch_loss / n_batches)

            self.fabric.print(f"[Epoch {epoch}] Train loss: {epoch_loss / n_batches:.4f}")

            if (epoch + 1) % self.check_val_every_n_epoch == 0:
                self.validate(epoch)

    def validate(self, epoch=None):
        self.model.eval()
        val_loader = tqdm(self.val_dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
        val_loss = 0.0
        n_loss_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                step_out = self.model.validation_step(batch, batch_idx)

                # Only accumulate loss if it exists
                if "loss" in step_out:
                    val_loss += step_out["loss"].item()
                    n_loss_batches += 1

                # Log all metrics
                self.fabric.log_dict(
                    {f"val/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_out.items()},
                    step=self.global_step,
                )

                self.global_step += 1
                val_loader.set_postfix(loss=(val_loss / n_loss_batches) if n_loss_batches > 0 else "N/A")

        if n_loss_batches > 0:
            avg_val_loss = val_loss / n_loss_batches
            self.fabric.print(f"[Epoch {epoch}] Validation loss: {avg_val_loss:.4f}")
        else:
            self.fabric.print(f"[Epoch {epoch}] Validation completed")

    def test(self, env=None, num_rollouts=None):
        if env is not None and num_rollouts is not None:
            test_dataset = GymDataset(env=env, num_rollouts=num_rollouts)
            test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_env, shuffle=False)
            test_loader = self.fabric.setup_dataloaders(test_loader)
        else:
            # Datamodule setup for train/val
            if hasattr(self.datamodule, "setup"):
                self.datamodule.setup(stage="test")
            test_loader = self.fabric.setup_dataloaders(self.datamodule.test_dataloader())

        self.model.eval()
        test_loader = tqdm(test_loader, desc="Testing", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                step_out = self.model.test_step(batch, batch_idx)

                # Log all metrics
                self.fabric.log_dict(
                    {f"test/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_out.items()},
                    step=self.global_step,
                )

                self.global_step += 1
                test_loader.set_postfix(loss=step_out.get("loss", "N/A"))

    def predict(self, *args, **kwargs):
        raise NotImplementedError
