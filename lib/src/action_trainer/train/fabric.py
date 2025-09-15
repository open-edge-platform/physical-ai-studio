# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Trainer with Fabric backend, heavily adapted from,
https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import lightning as L
import torch
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader
from tqdm import tqdm

from action_trainer.data.datamodules import collate_env
from action_trainer.data.gym import GymDataset
from action_trainer.train.utils import reformat_dataset_to_match_policy

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lightning.fabric.accelerators import Accelerator
    from lightning.fabric.loggers import Logger
    from lightning.fabric.strategies import Strategy

    from action_trainer.data import ActionDataModule
    from action_trainer.policies.base.base_lightning_module import ActionTrainerModule


class FabricActionTrainer:
    def __init__(
        self,
        datamodule: ActionDataModule,
        model: ActionTrainerModule,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: int | list[int] | str = "auto",
        precision: int | str = "32-true",
        plugins: Any | None = None,
        callbacks: list[Any] | Any | None = None,
        loggers: Logger | list[Logger] | None = None,
        max_epochs: int = 1000,
        max_steps: int | None = None,
        grad_accum_steps: int = 1,
        limit_train_batches: int | float = float("inf"),
        limit_val_batches: int | float = float("inf"),
        check_val_every_n_epoch: int = 1,
    ):
        self.model = model
        self.datamodule = datamodule
        reformat_dataset_to_match_policy(policy=self.model, datamodule=self.datamodule)
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

        self.global_step = 0
        self.current_epoch = 0
        self.grad_accum_steps = grad_accum_steps
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self._current_train_return: torch.Tensor | Mapping[str, Any] = {}
        self._current_val_return: torch.Tensor | Mapping[str, Any] | None = {}

        # Setup model + optimizer + dataloaders
        self.optimizer, self.scheduler_cfg = self._setup_model_and_optimizer()

        if hasattr(self.datamodule, "setup"):
            self.datamodule.setup(stage="fit")

        self.train_dataloader = self.fabric.setup_dataloaders(
            self.datamodule.train_dataloader(),
            use_distributed_sampler=True,
        )
        self.val_dataloader = self.fabric.setup_dataloaders(
            self.datamodule.val_dataloader(),
            use_distributed_sampler=True,
        )

    def _setup_model_and_optimizer(self):
        optimizer, scheduler_cfg = self._parse_optimizers_schedulers(self.model.configure_optimizers())
        model, optimizer = self.fabric.setup(self.model, optimizer)
        self.model = model
        return optimizer, scheduler_cfg

    def fit(self, ckpt_path: str | None = None):
        self.fabric.launch()

        # Resume from checkpoint if exists
        if ckpt_path is not None and Path(ckpt_path).is_dir():
            latest_checkpoint_path = self.get_latest_checkpoint(ckpt_path)
            if latest_checkpoint_path is not None:
                self.load(str(latest_checkpoint_path))
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        while not self.should_stop:
            self.train_loop()
            if self.should_validate:
                self.val_loop()
            self.step_scheduler(level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

        self.should_stop = False

    def test(self, env=None, num_rollouts=None):
        """
        Runs testing either from the datamodule or from a custom Gym environment.

        Args:
            env: Optional environment to generate a test dataset.
            num_rollouts: Number of rollouts to generate in the test dataset.
        """
        if env is not None and num_rollouts is not None:
            # Create a test dataset from the environment
            test_dataset = GymDataset(env=env, num_rollouts=num_rollouts)
            test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_env, shuffle=False)
            test_loader = self.fabric.setup_dataloaders(test_loader)
        else:
            # Use datamodule's test dataloader
            if not hasattr(self.datamodule, "test_dataloader"):
                msg = "No test dataloader found in datamodule."
                raise ValueError(msg)
            if hasattr(self.datamodule, "setup"):
                self.datamodule.setup(stage="test")
            test_loader = self.fabric.setup_dataloaders(self.datamodule.test_dataloader())

        self.model.eval()
        torch.set_grad_enabled(False)
        test_loader_iter = self.progbar_wrapper(test_loader, total=len(test_loader), desc="Testing")

        for batch_idx, batch in enumerate(test_loader_iter):
            step_out = self.model.test_step(batch, batch_idx)
            # detach tensors to prevent memory leaks
            step_out = apply_to_collection(step_out, torch.Tensor, lambda x: x.detach())
            # log metrics
            self.fabric.log_dict(
                {f"test/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in step_out.items()},
                step=self.global_step,
            )
            self.global_step += 1
            self._format_iterable(test_loader_iter, step_out, "test")

    @property
    def should_validate(self) -> bool:
        return self.current_epoch % self.check_val_every_n_epoch == 0

    def train_loop(self):
        self.fabric.call("on_train_epoch_start")
        iterable = self.progbar_wrapper(
            self.train_dataloader,
            total=min(len(self.train_dataloader), self.limit_train_batches),
            desc=f"Epoch {self.current_epoch} [Train]",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= self.limit_train_batches:
                break

            self.fabric.call("on_train_batch_start", batch, batch_idx)
            should_optim_step = self.global_step % self.grad_accum_steps == 0

            if should_optim_step:
                self.fabric.call("on_before_optimizer_step", self.optimizer)
                self.optimizer.step(partial(self.training_step, batch=batch, batch_idx=batch_idx))
                self.fabric.call("on_before_zero_grad", self.optimizer)
                self.optimizer.zero_grad()
            else:
                self.training_step(batch=batch, batch_idx=batch_idx)

            self.fabric.call("on_train_batch_end", self._current_train_return, batch, batch_idx)
            if should_optim_step:
                self.step_scheduler(level="step", current_value=self.global_step)

            self._format_iterable(iterable, self._current_train_return, "train")
            self.global_step += int(should_optim_step)
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        self.fabric.call("on_train_epoch_end")

    def val_loop(self):
        if self.val_dataloader is None:
            return
        if not is_overridden("validation_step", _unwrap_objects(self.model)):
            L.fabric.utilities.rank_zero_warn("No validation_step implemented, skipping validation.")
            return

        self.model.eval()
        torch.set_grad_enabled(False)
        self.fabric.call("on_validation_epoch_start")

        iterable = self.progbar_wrapper(
            self.val_dataloader,
            total=min(len(self.val_dataloader), self.limit_val_batches),
            desc=f"Epoch {self.current_epoch} [Val]",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= self.limit_val_batches:
                break
            self.fabric.call("on_validation_batch_start", batch, batch_idx)
            out = self.model.validation_step(batch, batch_idx)
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())
            self._current_val_return = out
            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._format_iterable(iterable, out, "val")

        self.fabric.call("on_validation_epoch_end")
        self.model.train()
        torch.set_grad_enabled(True)

    def training_step(self, batch, batch_idx):
        outputs = self.model.training_step(batch, batch_idx)
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        self.fabric.call("on_before_backward", loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")
        self._current_train_return = apply_to_collection(outputs, torch.Tensor, lambda x: x.detach())
        return loss

    def step_scheduler(self, level: Literal["step", "epoch"], current_value: int):
        if self.scheduler_cfg is None:
            return
        if self.scheduler_cfg["interval"] != level:
            return
        if current_value % self.scheduler_cfg["frequency"] != 0:
            return
        # monitor value
        monitor_val = None
        if isinstance(self._current_val_return, dict):
            monitor_val = self._current_val_return.get("loss", None)
        self.model.lr_scheduler_step(self.scheduler_cfg["scheduler"], monitor_val)

    @staticmethod
    def progbar_wrapper(iterable: Iterable, total: int, **kwargs):
        if L.Fabric().is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, path: str):
        state = self.fabric.load(path)
        self.global_step = state.pop("global_step", 0)
        self.current_epoch = state.pop("current_epoch", 0)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Path | None:
        if not Path(checkpoint_dir).is_dir():
            return None
        items = sorted(Path(checkpoint_dir).iterdir())
        if not items:
            return None
        return Path(checkpoint_dir).joinpath(items[-1])

    def _parse_optimizers_schedulers(self, configure_optim_output):
        # similar parsing logic as MyCustomTrainer
        if isinstance(configure_optim_output, torch.optim.Optimizer):
            return configure_optim_output, None
        if isinstance(configure_optim_output, (list, tuple)):
            return configure_optim_output[0], None
        return configure_optim_output, None

    @staticmethod
    def _format_iterable(prog_bar, candidates, prefix: str):
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            if isinstance(candidates, dict):
                for k, v in candidates.items():
                    # Handle tensor, scalar, list
                    if isinstance(v, torch.Tensor):
                        val_str = f"{v.item():.3f}"
                    elif isinstance(v, (float, int)):
                        val_str = f"{v:.3f}"
                    elif isinstance(v, list):
                        val_str = f"{v}"  # just str() the list
                    else:
                        val_str = str(v)
                    postfix_str += f" {prefix}_{k}: {val_str}"
            elif isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {candidates.item():.3f}"
            elif isinstance(candidates, list):
                postfix_str += f" {prefix}: {candidates}"
            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
