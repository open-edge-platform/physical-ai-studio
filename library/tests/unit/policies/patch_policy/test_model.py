# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Encoder, head, loss, and prediction wiring with small CPU models."""

from dataclasses import replace

import pytest
import torch

from physicalai.data import Feature, FeatureType
from physicalai.policies.patch_policy.components.heads import BehaviourTransformerHead, DiffusionHead
from physicalai.policies.patch_policy.model import PatchPolicyModel


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_model_computes_loss_gradients_and_full_action_chunk(tiny_encoder, small_config, model_batch, action_head):
    config = replace(small_config, action_head=action_head)
    model = PatchPolicyModel.from_config(config)
    if isinstance(model.action_head, BehaviourTransformerHead):
        model.action_head.quantizer.fitted.fill_(True)

    loss, metrics = model(model_batch)
    loss.backward()
    model.eval()
    actions = model(model_batch)

    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert not metrics["loss"].requires_grad
    assert any(parameter.grad is not None for parameter in model.action_head.parameters())
    assert model.encoder.weight.grad is None
    assert actions.shape == (1, 2, 2)
    assert torch.isfinite(actions).all()
    assert isinstance(model.action_head, DiffusionHead if action_head == "diffusion" else BehaviourTransformerHead)


def test_context_preserves_time_camera_and_patch_order(tiny_encoder, small_config):
    config = replace(
        small_config,
        input_features=[
            Feature(name="right", ftype=FeatureType.VISUAL, shape=(3, 4, 4)),
            Feature(name="left", ftype=FeatureType.VISUAL, shape=(3, 4, 4)),
        ],
    )
    model = PatchPolicyModel.from_config(config)
    left = torch.tensor([20.0, 40.0]).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 4, 4)
    right = torch.tensor([10.0, 30.0]).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 4, 4)

    nested = model._context({"images": {"left": left, "right": right}})["tokens"]
    flattened = model._context({"images.left": left, "images.right": right})["tokens"]

    assert model.camera_names == ("right", "left")
    assert nested.shape == (1, 8, 4)
    torch.testing.assert_close(nested[0, :, 0], torch.tensor([10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]))
    torch.testing.assert_close(flattened, nested)


@pytest.mark.parametrize(
    ("images", "error", "message"),
    [
        ({}, KeyError, "Missing tensor"),
        ({"front": torch.zeros(1, 3, 4, 4)}, ValueError, "must have shape"),
        ({"front": torch.zeros(1, 1, 3, 4, 4)}, ValueError, "observation steps"),
        ({"front": torch.zeros(1, 2, 3, 4, 4), "extra": torch.zeros(1)}, ValueError, "unconfigured"),
        (torch.zeros(1), TypeError, "dictionary"),
    ],
)
def test_model_rejects_invalid_camera_inputs(tiny_encoder, small_config, images, error, message):
    model = PatchPolicyModel.from_config(small_config)

    with pytest.raises(error, match=message):
        model._context({"images": images})


def test_model_masks_padded_diffusion_losses(tiny_encoder, small_config, model_batch, monkeypatch):
    model = PatchPolicyModel.from_config(small_config)
    losses = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]], requires_grad=True)
    monkeypatch.setattr(model.action_head, "compute_loss", lambda actions, context: losses)
    model_batch["extra.action_is_pad"] = torch.tensor([[False, False, True]])

    loss, _ = model.compute_loss(model_batch)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(losses.grad[:, 2], torch.zeros(1, 2))


def test_validation_uses_current_chunk_and_ignores_padding(tiny_encoder, small_config, model_batch, monkeypatch):
    model = PatchPolicyModel.from_config(small_config)
    model_batch["action"] = torch.tensor([[[100.0, 100.0], [1.0, 1.0], [20.0, 20.0]]])
    model_batch["extra.action_is_pad"] = torch.tensor([[False, False, True]])
    monkeypatch.setattr(model.ema_action_head, "sample", lambda context: torch.zeros(1, 2, 2))

    loss, metrics = model.compute_val_loss(model_batch)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(metrics["action_mse"], loss)


def test_prediction_uses_ema_instead_of_training_head(tiny_encoder, small_config, model_batch, monkeypatch):
    model = PatchPolicyModel.from_config(small_config)
    monkeypatch.setattr(model.action_head, "sample", lambda context: torch.zeros(1, 2, 2))
    monkeypatch.setattr(model.ema_action_head, "sample", lambda context: torch.ones(1, 2, 2))

    actions = model.predict_action_chunk(model_batch)

    torch.testing.assert_close(actions, torch.ones(1, 2, 2))


def test_ema_copies_then_averages_weights_and_stays_frozen(tiny_encoder, small_config):
    model = PatchPolicyModel.from_config(small_config)
    current = next(model.action_head.parameters())
    averaged = next(model.ema_action_head.parameters())
    with torch.no_grad():
        current.fill_(2.0)
        averaged.zero_()
    model.update_ema()
    torch.testing.assert_close(averaged, torch.full_like(averaged, 2.0))

    with torch.no_grad():
        current.fill_(4.0)
    model.update_ema()
    with torch.no_grad():
        current.fill_(6.0)
    model.update_ema()
    model.train()

    decay = 1 - pow(2, -0.75)
    expected = 4.0 * decay + 6.0 * (1 - decay)
    torch.testing.assert_close(averaged, torch.full_like(averaged, expected))
    assert model.ema_optimization_step.item() == 3
    assert not model.ema_action_head.training
    assert all(not parameter.requires_grad for parameter in model.ema_action_head.parameters())


def test_temporal_indices_cover_history_and_action_targets(tiny_encoder, small_config):
    model = PatchPolicyModel.from_config(small_config)

    assert model.observation_delta_indices == [-1, 0]
    assert model.action_delta_indices == [-1, 0, 1]
    assert model.reward_delta_indices is None
