# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise diffusion EMA timing through real Lightning optimizer updates."""

import lightning
import pytest
import torch
from dataclasses import replace
from lightning.pytorch.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from physicalai.data import Dataset, Observation
from physicalai.policies.patch_policy.policy import PatchPolicy


class TrainingDataset(Dataset):
    """Five prebatched observations, including a partial accumulation window."""

    def __init__(self, config):
        self.config = config
        self._delta_indices = {}

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return Observation(
            images={"front": torch.full((1, 2, 3, 4, 4), 0.25)},
            action=torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]),
        )

    raw_features = property(lambda self: {})
    observation_features = property(lambda self: {feature.name: feature for feature in self.config.input_features})
    action_features = property(lambda self: {feature.name: feature for feature in self.config.output_features})
    fps = property(lambda self: 10)
    tolerance_s = property(lambda self: 0.01)
    delta_indices = property(
        lambda self: self._delta_indices,
        lambda self, values: setattr(self, "_delta_indices", values),
    )


class TrainingDataModule(lightning.LightningDataModule):
    """Expose the metadata and Observation batches required by Policy.setup."""

    def __init__(self, config):
        super().__init__()
        self.train_dataset = TrainingDataset(config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None)


class CheckEMA(lightning.Callback):
    """Check every batch against the last actual optimizer update."""

    def __init__(self):
        self.updates = 0
        self.previous_weights = None
        self.expected_ema = None

    def on_train_start(self, trainer, policy):
        assert int(policy.model.ema_optimization_step) == 0
        self.previous_weights = [parameter.detach().clone() for parameter in policy.model.action_head.parameters()]
        self.expected_ema = [parameter.detach().clone() for parameter in policy.model.ema_action_head.parameters()]

    def on_train_batch_end(self, trainer, policy, outputs, batch, batch_idx):
        current = list(policy.model.action_head.parameters())
        updated = any(
            not torch.equal(before, after) for before, after in zip(self.previous_weights, current, strict=True)
        )
        if updated:
            step = max(0, self.updates - 1)
            decay = min(1 - (1 + step) ** -0.75, 0.9999) if step > 0 else 0.0
            self.expected_ema = [
                before * decay + after.detach() * (1 - decay)
                for before, after in zip(self.expected_ema, current, strict=True)
            ]
            self.updates += 1
        assert int(policy.model.ema_optimization_step) == self.updates
        for expected, actual in zip(self.expected_ema, policy.model.ema_action_head.parameters(), strict=True):
            torch.testing.assert_close(actual, expected)
        self.previous_weights = [parameter.detach().clone() for parameter in current]


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_ema_follows_actual_optimizer_updates(tiny_encoder, small_config, accumulate_grad_batches):
    policy = PatchPolicy.from_config(small_config, compile_model=False)
    check = CheckEMA()
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[check],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(policy, datamodule=TrainingDataModule(policy.config))

    expected_updates = (5 + accumulate_grad_batches - 1) // accumulate_grad_batches
    assert check.updates == expected_updates
    assert int(policy.model.ema_optimization_step) == expected_updates
    assert trainer.global_step == expected_updates


def test_ema_ignores_grad_scaler_skipped_step(tiny_encoder, small_config):
    policy = PatchPolicy.from_config(small_config, compile_model=False)
    check = CheckEMA()

    class OverflowFirstStep(lightning.Callback):
        def on_before_backward(self, trainer, policy, loss):
            if trainer.global_step == 0:
                # Inject overflow before GradScaler unscales and checks gradients.
                loss.register_hook(lambda gradient: gradient * float("inf"))

    # A real CPU GradScaler exercises skip detection without requiring CUDA.
    precision = MixedPrecision("16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu", init_scale=8.0))
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        plugins=[precision],
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[OverflowFirstStep(), check],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )

    trainer.fit(policy, datamodule=TrainingDataModule(policy.config))

    assert trainer.global_step == 2  # Lightning counts attempted steps, including overflow.
    assert check.updates == 1
    assert int(policy.model.ema_optimization_step) == 1


def test_loaded_policy_resumes_full_config_and_optimizer_state(tiny_encoder, small_config, tmp_path):
    config = replace(small_config, dropout=0.1, embedding_dropout=0.2)
    policy = PatchPolicy.from_config(config, compile_model=False)
    settings = {
        "accelerator": "cpu", "devices": 1, "limit_train_batches": 1,
        "logger": False, "enable_checkpointing": False, "enable_progress_bar": False,
        "enable_model_summary": False, "limit_val_batches": 0, "num_sanity_val_steps": 0,
    }
    trainer = lightning.Trainer(max_epochs=1, **settings)
    trainer.fit(policy, datamodule=TrainingDataModule(policy.config))
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)

    restored = PatchPolicy.load_from_checkpoint(path, compile_model=False)
    resumed = lightning.Trainer(max_epochs=2, **settings)
    resumed.fit(restored, datamodule=TrainingDataModule(restored.config), ckpt_path=path)

    assert restored.config == policy.config
    assert restored.config.dropout == 0.1
    assert restored.config.embedding_dropout == 0.2
    assert resumed.global_step == 2
    assert int(restored.model.ema_optimization_step) == 2
    assert all(int(state["step"]) == 2 for state in resumed.optimizers[0].state.values())
