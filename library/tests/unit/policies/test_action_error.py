# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the validation action error in dataset units."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from physicalai.data.observation import Observation
from physicalai.policies.base.policy import Policy


class _FixedPolicy(Policy):
    """Predicts a fixed chunk and returns it from compute_val_loss when asked to."""

    def __init__(self, prediction: torch.Tensor, *, return_pred: bool, n_action_steps: int) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self.prediction = prediction
        self.return_pred = return_pred
        self.predict_calls = 0

    def forward(self, batch: Observation) -> torch.Tensor:
        return self.predict_action_chunk(batch)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:  # noqa: ARG002
        self.predict_calls += 1
        return self.prediction

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:  # noqa: ARG002
        loss = torch.tensor(0.0)
        loss_dict: dict[str, torch.Tensor | float] = {"loss": loss}
        if self.return_pred:
            loss_dict["action_pred"] = self.prediction
        return loss, loss_dict


def _run(policy: Policy, batch: Observation, names: object = None) -> dict[str, float]:
    logged: dict[str, float] = {}
    datamodule = SimpleNamespace(train_dataset=SimpleNamespace(raw_features={"action": {"names": names}}))
    with (
        patch.object(Policy, "trainer", SimpleNamespace(datamodule=datamodule)),
        patch.object(policy, "log"),
        patch.object(policy, "log_dict", side_effect=lambda d, **_: logged.update({k: float(v) for k, v in d.items()})),
    ):
        policy.validation_step(batch, 0)
    return logged


# Two samples, chunk of 3, two joints. Ground truth is zero, so error == |prediction|.
_PRED = torch.tensor(
    [
        [[1.0, 10.0], [2.0, 20.0], [100.0, 100.0]],
        [[3.0, 30.0], [100.0, 100.0], [100.0, 100.0]],
    ],
)


@pytest.mark.parametrize("return_pred", [True, False])
def test_scores_executed_unpadded_steps_per_joint(return_pred: bool) -> None:
    policy = _FixedPolicy(_PRED, return_pred=return_pred, n_action_steps=2)
    # Step 3 is beyond n_action_steps; sample 2's step 2 is past the episode end.
    batch = Observation(
        action=torch.zeros(2, 3, 2),
        extra={"action_is_pad": torch.tensor([[False, False, False], [False, True, True]])},
    )

    logged = _run(policy, batch, names=["shoulder.pos", "gripper.pos"])

    assert logged["val/action_error/shoulder.pos"] == pytest.approx(2.0)  # (1 + 2 + 3) / 3
    assert logged["val/action_error/gripper.pos"] == pytest.approx(20.0)
    assert logged["val/action_error"] == pytest.approx(11.0)
    assert policy.predict_calls == (0 if return_pred else 1)


@pytest.mark.parametrize("names", [None, ["only_one"], {"motors": ["a", "b", "c"]}])
def test_falls_back_to_indexed_names(names: object) -> None:
    policy = _FixedPolicy(_PRED, return_pred=True, n_action_steps=1)

    logged = _run(policy, Observation(action=torch.zeros(2, 3, 2)), names=names)

    assert set(logged) == {"val/action_error", "val/action_error/joint_0", "val/action_error/joint_1"}


def test_legacy_dict_names_are_used() -> None:
    policy = _FixedPolicy(_PRED, return_pred=True, n_action_steps=1)

    logged = _run(policy, Observation(action=torch.zeros(2, 3, 2)), names={"motors": ["a", "b"]})

    assert logged["val/action_error/a"] == pytest.approx(2.0)


def test_skipped_without_chunked_actions() -> None:
    policy = _FixedPolicy(_PRED, return_pred=False, n_action_steps=1)

    assert _run(policy, Observation(state=torch.zeros(2, 4))) == {}
    assert policy.predict_calls == 0
