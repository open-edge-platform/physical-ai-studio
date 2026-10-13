# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiny seeded CPU parity check for the LeRobot progress wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        error = self.linear(batch["input"]) - batch["target"]
        return error.square().mean(), None


class _CPUAccelerator:
    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def unwrap_model(self, model: nn.Module, *, keep_fp32_wrapper: bool) -> nn.Module:  # noqa: ARG002
        return model


class _RecordingHooks:
    def __init__(self) -> None:
        self.steps: list[tuple[int, int | None, Mapping[str, float]]] = []

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        self.steps.append((step, total, dict(metrics)))

    def should_stop(self) -> bool:
        return False


def _metrics_tracker(initial_step: int = 0):
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    meters = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")}
    return MetricsTracker(1, 4, 1, meters, initial_step=initial_step)


@pytest.mark.integration
def test_engine_progress_wrapper_preserves_seeded_native_cpu_loss_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Engine hook adds progress reporting without changing LeRobot's CPU update."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train
    from lerobot.utils import import_utils
    from physicalai.engine import Engine

    torch.manual_seed(31415)
    baseline_policy = _TinyPolicy()
    engine_policy = _TinyPolicy()
    engine_policy.load_state_dict(baseline_policy.state_dict())
    baseline_optimizer = torch.optim.SGD(baseline_policy.parameters(), lr=0.05)
    engine_optimizer = torch.optim.SGD(engine_policy.parameters(), lr=0.05)
    batch = {
        "input": torch.tensor([[0.25, -0.5], [1.0, 0.75]], dtype=torch.float32),
        "target": torch.tensor([[0.2], [-0.1]], dtype=torch.float32),
    }
    accelerator = _CPUAccelerator()
    native_update = lerobot_train.update_policy
    baseline_tracker = _metrics_tracker()
    native_update(
        baseline_tracker,
        baseline_policy,
        batch,
        baseline_optimizer,
        0.0,
        accelerator=accelerator,
    )

    hooks = _RecordingHooks()
    engine_tracker = _metrics_tracker()

    def train_with_engine_hook(config: TrainPipelineConfig) -> None:  # noqa: ARG001
        lerobot_train.update_policy(
            engine_tracker,
            engine_policy,
            batch,
            engine_optimizer,
            0.0,
            accelerator=accelerator,
        )

    monkeypatch.setattr(lerobot_train, "train", train_with_engine_hook)
    monkeypatch.setattr(import_utils, "register_third_party_plugins", lambda: None)
    config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/pusht"), steps=1)

    Engine().train(config, hooks=hooks)

    torch.testing.assert_close(engine_tracker.loss.val, baseline_tracker.loss.val, rtol=1e-6, atol=1e-7)
    for name, baseline_parameter in baseline_policy.state_dict().items():
        torch.testing.assert_close(engine_policy.state_dict()[name], baseline_parameter, rtol=1e-6, atol=1e-7)
    assert len(hooks.steps) == 1
    assert hooks.steps[0][0:2] == (1, 1)
    assert hooks.steps[0][2]["loss"] == pytest.approx(baseline_tracker.loss.val)
