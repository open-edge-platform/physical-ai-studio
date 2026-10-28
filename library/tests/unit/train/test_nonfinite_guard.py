# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Non-finite training values must fail before corrupting model weights."""

from unittest.mock import MagicMock

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from physicalai.train.callbacks import NonFiniteGuard


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_rejected(value: float) -> None:
    trainer = MagicMock(spec=L.Trainer)
    trainer.global_step = 5
    with pytest.raises(FloatingPointError, match="step 5"):
        NonFiniteGuard().on_before_backward(trainer, MagicMock(), torch.tensor(value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradient_is_rejected(value: float) -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([value])
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(RuntimeError, match="non-finite"):
        NonFiniteGuard().on_before_optimizer_step(MagicMock(), MagicMock(), optimizer)
    torch.testing.assert_close(parameter, torch.zeros(1))


def test_finite_loss_and_gradients_are_unchanged() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    guard = NonFiniteGuard()
    guard.on_before_backward(MagicMock(), MagicMock(), torch.tensor(2.0))
    guard.on_before_optimizer_step(MagicMock(), MagicMock(), optimizer)
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]))
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.7, 0.6]))


def test_lightning_rejects_infinite_gradient_from_finite_loss() -> None:
    class ExplodingGradient(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            del batch, batch_idx
            return self.weight.sqrt()

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.1)

    model = ExplodingGradient()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[NonFiniteGuard()],
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(model, train_dataloaders=DataLoader([torch.zeros(1)]))
    assert trainer.global_step == 0
    torch.testing.assert_close(model.weight, torch.zeros(()))
