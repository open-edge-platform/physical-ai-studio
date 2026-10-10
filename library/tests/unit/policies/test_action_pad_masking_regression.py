# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-policy regression safety net for end-of-episode action padding.

LeRobot clamps action-chunk queries at episode boundaries by repeating the
terminal action and flagging the clamped steps as ``action_is_pad``. Every
chunked-action policy must exclude those steps from its training loss, or the
tail of every episode silently trains the policy towards a frozen pose.

Per-policy tests already pin the exact masking arithmetic (see
``test_pi05.py::TestActionPaddingMask`` and
``test_smolvla.py::TestActionPaddingMask``). This module instead guards
against the *pattern* being forgotten when a new chunked policy is added, or
quietly dropped from an existing one: for every policy listed in
``MASKING_POLICIES`` below, flipping ``action_is_pad`` from "nothing padded"
to "tail padded" must actually change the reported loss.

Note:
    This regression suite covers the first-party chunked-action policies that
    currently implement padding masking. LeRobot wrappers are exercised in
    their own wrapper/equivalence suites rather than through these direct
    model stubs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from transformers import BatchFeature

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, EXTRA, IMAGES
from physicalai.policies.act.model import ACT
from physicalai.policies.pi05.model import Pi05Model
from physicalai.policies.rldx1.model import RLDXActionModel, Rldx1Model
from physicalai.policies.smolvla.model import SmolVLAModel

BATCH, CHUNK, ACTION_DIM = 2, 6, 4


def _pad_mask(all_padded_from: int | None) -> torch.Tensor:
    """Build a ``(BATCH, CHUNK)`` bool mask, padded from a given step onward."""
    mask = torch.zeros(BATCH, CHUNK, dtype=torch.bool)
    if all_padded_from is not None:
        mask[:, all_padded_from:] = True
    return mask


def _per_step_error() -> torch.Tensor:
    """A ``(BATCH, CHUNK, ACTION_DIM)`` error that grows with the chunk step.

    Uniform error would make masked and unmasked means coincide by accident;
    growing error ties the test to the tail actually being excluded.
    """
    step = torch.arange(1, CHUNK + 1, dtype=torch.float32).view(1, CHUNK, 1)
    return step.expand(BATCH, CHUNK, ACTION_DIM).clone()


def _act_loss(action_is_pad: torch.Tensor, *, validation: bool = False) -> float:
    """Compute ACT loss with a per-step-varying prediction error."""
    actions = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    actions_hat = _per_step_error()
    batch: dict[str, Any] = {
        ACTION: actions,
        EXTRA + ".action_is_pad": action_is_pad,
    }

    class ModelStub:
        def __call__(self, _batch: dict[str, Any]) -> tuple[torch.Tensor, tuple[None, None]]:
            return actions_hat, (None, None)

        def train(self) -> None:
            pass

        def eval(self) -> None:
            pass

    stub = SimpleNamespace(
        _input_normalizer=lambda b: b,
        _model=ModelStub(),
        _config=SimpleNamespace(use_vae=False, kl_weight=0.0),
    )
    compute = ACT.compute_val_loss if validation else ACT.compute_loss
    loss, _ = compute(stub, batch)
    return float(loss)


def _pi05_loss(action_is_pad: torch.Tensor) -> float:
    """Compute ``Pi05Model._flow_matching_loss`` with a per-step-varying error.

    The stub pins noise and ground-truth action to zero, so the flow-matching
    target ``u_t`` is zero and the per-element loss collapses to
    ``_predict_velocity(...) ** 2``. Feeding back ``sqrt(error)`` therefore
    reproduces ``error`` exactly (see the equivalent stub in
    ``test_pi05.py::TestActionPaddingMask``).
    """
    velocity = _per_step_error().sqrt()
    batch: dict[str, Any] = {
        IMAGES: None,
        IMAGE_MASKS: None,
        TOKENIZED_PROMPT: None,
        TOKENIZED_PROMPT_MASK: None,
        ACTION: torch.zeros(BATCH, CHUNK, ACTION_DIM),
        EXTRA + ".action_is_pad": action_is_pad,
    }
    stub = SimpleNamespace(
        _snapflow_enabled=False,
        _dataset_stats={ACTION: {"shape": (ACTION_DIM,)}},
        sample_noise=lambda shape, _device: torch.zeros(shape),
        sample_time=lambda b, _device: torch.full((b,), 0.5),
        embed_prefix=lambda *_a, **_kw: (None, None, None),
        _predict_velocity=lambda *_a, **_kw: velocity,
    )
    loss, _ = Pi05Model._flow_matching_loss(stub, batch)
    return float(loss)


def _smolvla_loss(action_is_pad: torch.Tensor) -> float:
    """Compute ``SmolVLAModel.compute_loss`` with a per-step-varying error."""
    losses = _per_step_error()
    batch: dict[str, Any] = {
        IMAGES: None,
        IMAGE_MASKS: None,
        TOKENIZED_PROMPT: None,
        TOKENIZED_PROMPT_MASK: None,
        EXTRA + ".action_is_pad": action_is_pad,
    }
    stub = SimpleNamespace(
        _preprocess_batch=lambda b: b,
        _prepare_state=lambda b: None,
        _prepare_action=lambda b: None,
        _model=SimpleNamespace(forward=lambda *_a, **_kw: (losses.clone(), None)),
        _dataset_stats={ACTION: {"shape": (ACTION_DIM,)}},
    )
    loss, _ = SmolVLAModel.compute_loss(stub, batch)
    return float(loss)


def _rldx1_action_mask(action_is_pad: torch.Tensor) -> torch.Tensor:
    """Expand an episode-bound mask over the RLDX-1 action dimension."""
    return (~action_is_pad).unsqueeze(-1).expand(BATCH, CHUNK, ACTION_DIM).to(torch.float32)


def _rldx1_train_loss(action_is_pad: torch.Tensor) -> float:
    """Compute RLDX-1's training loss with lightweight action-model stubs."""
    actions = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    predicted_velocity = _per_step_error().sqrt()
    backbone_output = BatchFeature(data={"backbone_features": torch.zeros(BATCH, 1, 1)})
    action_input = BatchFeature(
        data={
            "state": torch.zeros(BATCH, 1, 1),
            ACTION: actions,
            "embodiment_id": torch.zeros(BATCH, dtype=torch.long),
            "action_mask": _rldx1_action_mask(action_is_pad),
        },
    )
    stub = SimpleNamespace(
        process_backbone_output=lambda output: output,
        state_encoder=lambda *_args: torch.zeros(BATCH, 1, 1),
        state_dropout_prob=0.0,
        training=True,
        state_additive_noise_scale=0.0,
        sample_time=lambda batch_size, **_kwargs: torch.zeros(batch_size),
        action_horizon=CHUNK,
        action_encoder=lambda *_args: torch.zeros(BATCH, CHUNK, 1),
        add_pos_embed=False,
        model=lambda **_kwargs: torch.zeros(BATCH, CHUNK + 1, 1),
        action_decoder=lambda *_args: predicted_velocity,
        physics=SimpleNamespace(
            prepare_train=lambda *_args: (None, None, None),
            compute_loss=lambda *_args: None,
        ),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "randn", lambda *_args, **_kwargs: torch.zeros_like(actions))
        outputs = RLDXActionModel.forward(stub, backbone_output, action_input)
    return float(outputs["loss"])


def _rldx1_val_loss(action_is_pad: torch.Tensor) -> float:
    """Compute RLDX-1's validation loss with a fixed prediction error."""
    target = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    predicted = _per_step_error().sqrt()
    batch = {
        ACTION: target,
        "action_mask": _rldx1_action_mask(action_is_pad),
    }
    stub = SimpleNamespace(get_action=lambda _batch: predicted)

    loss, _ = Rldx1Model.compute_val_loss(stub, batch)
    return float(loss)


# Maps a policy name to a callable that computes its training loss for a
# given ``action_is_pad`` mask, holding the underlying prediction error fixed.
MASKING_POLICIES: dict[str, Callable[[torch.Tensor], float]] = {
    "act": _act_loss,
    "pi05": _pi05_loss,
    "rldx1": _rldx1_train_loss,
    "smolvla": _smolvla_loss,
}


@pytest.mark.parametrize("policy_name", sorted(MASKING_POLICIES))
def test_action_is_pad_actually_affects_loss(policy_name: str) -> None:
    """Flipping ``action_is_pad`` must change the reported loss.

    This is a safety net, not a correctness check (see the per-policy
    ``TestActionPaddingMask`` suites for that): if a future edit removes the
    masking call or a new chunked policy forgets to add it, this test fails
    because the loss becomes insensitive to which steps are padding.
    """
    compute_loss = MASKING_POLICIES[policy_name]

    no_pad = _pad_mask(all_padded_from=None)
    tail_pad = _pad_mask(all_padded_from=CHUNK // 2)

    loss_no_pad = compute_loss(no_pad)
    loss_tail_pad = compute_loss(tail_pad)

    assert loss_no_pad != pytest.approx(loss_tail_pad), (
        f"{policy_name}: loss is insensitive to action_is_pad — padded steps are not masked"
    )


@pytest.mark.parametrize("policy_name", sorted(MASKING_POLICIES))
def test_fully_unpadded_batch_uses_the_full_error(policy_name: str) -> None:
    """With nothing padded, the loss must reflect every step's error.

    Guards against a masking bug that accidentally zeroes out valid steps
    too (e.g. an inverted mask), which `test_action_is_pad_actually_affects_loss`
    alone would not catch.
    """
    compute_loss = MASKING_POLICIES[policy_name]

    no_pad = _pad_mask(all_padded_from=None)
    expected = float(_per_step_error().mean())
    assert compute_loss(no_pad) == pytest.approx(expected)


@pytest.mark.parametrize("stage", ["train", "val"])
def test_act_padding_does_not_dilute_loss(stage: str) -> None:
    """ACT loss should average valid steps only, like Pi0.5 and SmolVLA."""
    no_pad = _pad_mask(all_padded_from=None)
    tail_pad = _pad_mask(all_padded_from=CHUNK // 2)
    validation = stage == "val"

    assert _act_loss(no_pad, validation=validation) == pytest.approx(3.5)
    assert _act_loss(tail_pad, validation=validation) == pytest.approx(2.0)


@pytest.mark.parametrize("stage", ["train", "val"])
def test_rldx1_padding_does_not_dilute_loss(stage: str) -> None:
    """RLDX-1 loss should average over valid action elements only."""
    no_pad = _pad_mask(all_padded_from=None)
    tail_pad = _pad_mask(all_padded_from=CHUNK // 2)
    compute_loss = _rldx1_train_loss if stage == "train" else _rldx1_val_loss

    assert compute_loss(no_pad) == pytest.approx(3.5)
    assert compute_loss(tail_pad) == pytest.approx(2.0)
