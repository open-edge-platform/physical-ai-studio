# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native MolmoAct2 model."""

import pytest
import torch

from physicalai.data.observation import ACTION
from physicalai.policies.molmoact2 import MolmoAct2Config, MolmoAct2Model
from physicalai.policies.molmoact2.components import ActionExpert, MolmoAct2ForConditionalGeneration
from physicalai.policies.molmoact2.components.backbone import _merge_image_features
from physicalai.policies.molmoact2.model import _masked_action_mse


@pytest.fixture
def model(tiny_molmoact2_config: MolmoAct2Config) -> MolmoAct2Model:
    return MolmoAct2Model.from_config(tiny_molmoact2_config)


def test_model_assembly_and_checkpoint_keys(model: MolmoAct2Model) -> None:
    assert isinstance(model.backbone, MolmoAct2ForConditionalGeneration)
    assert isinstance(model.backbone.model.action_expert, ActionExpert)
    assert "backbone.lm_head.weight" in model.state_dict()
    assert any(name.startswith("backbone.model.transformer.") for name in model.state_dict())
    assert not hasattr(model, "config")


def test_merge_image_features_matches_compact_update_with_per_example_padding() -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    raw_image_features = torch.tensor(
        [
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [100.0, 100.0, 100.0]],
            [[70.0, 80.0, 90.0], [200.0, 200.0, 200.0], [300.0, 300.0, 300.0]],
        ],
        requires_grad=True,
    )
    valid_token = torch.tensor([[True, True, False], [True, False, False]])
    image_features = torch.where(
        valid_token[..., None],
        raw_image_features,
        torch.zeros_like(raw_image_features),
    )
    is_image_patch = torch.tensor([[False, True, False, True], [True, False, False, False]])
    expected = embeddings.detach().clone().reshape(-1, 3)
    expected[is_image_patch.flatten()] += raw_image_features.detach()[valid_token]
    expected = expected.reshape_as(embeddings)

    merged = _merge_image_features(embeddings, image_features, is_image_patch)

    torch.testing.assert_close(merged, expected)
    merged.sum().backward()
    torch.testing.assert_close(embeddings.grad, torch.ones_like(embeddings))
    expected_image_grad = valid_token[..., None].expand_as(raw_image_features).to(raw_image_features.dtype)
    torch.testing.assert_close(raw_image_features.grad, expected_image_grad)


def test_masked_action_mse_excludes_padding_and_preserves_gradients() -> None:
    predicted = torch.tensor([[[[2.0, 100.0], [50.0, 50.0]]]], requires_grad=True)
    loss = _masked_action_mse(
        predicted,
        torch.zeros_like(predicted),
        action_horizon_is_pad=torch.tensor([[False, True]]),
        action_dim_is_pad=torch.tensor([[False, True]]),
    )

    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([[[[4.0, 0.0], [0.0, 0.0]]]]))


def test_forward_dispatches_by_mode(model: MolmoAct2Model, monkeypatch: pytest.MonkeyPatch) -> None:
    loss = torch.tensor(1.0)
    actions = torch.ones(1, 2, 4)
    monkeypatch.setattr(model, "compute_loss", lambda _: (loss, {"loss": loss}))
    monkeypatch.setattr(model, "predict_action_chunk", lambda _: actions)

    model.train()
    assert model({})[0] is loss
    model.eval()
    assert model({}) is actions


def test_predict_action_chunk_trims_output(
    model: MolmoAct2Model,
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = torch.ones(1, tiny_molmoact2_config.chunk_size, tiny_molmoact2_config.max_action_dim)
    calls: dict[str, object] = {}

    def generate(**kwargs: object) -> torch.Tensor:
        calls.update(kwargs)
        return generated

    monkeypatch.setattr(model._unwrapped_backbone.model, "generate_actions_from_inputs", generate)
    actions = model.predict_action_chunk({"input_ids": torch.zeros(1, 1, dtype=torch.long)})

    assert actions.shape == (1, tiny_molmoact2_config.n_action_steps, 4)
    assert calls["action_horizon"] == tiny_molmoact2_config.chunk_size


def test_validation_reports_action_and_flow_losses(
    model: MolmoAct2Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicted = torch.ones(1, 2, 4)
    target = torch.zeros(1, 4, 4)
    flow_loss = torch.tensor(3.0)
    monkeypatch.setattr(model, "predict_action_chunk", lambda *_, **__: predicted)
    monkeypatch.setattr(model, "compute_loss", lambda _: (flow_loss, {"loss": flow_loss}))

    loss, metrics = model.compute_val_loss({ACTION: target})

    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(metrics["action_mse"], loss)
    torch.testing.assert_close(metrics["action_flow_loss"], flow_loss)


def test_gradient_checkpointing_and_freezing(model: MolmoAct2Model) -> None:
    backbone = model._unwrapped_backbone.model
    model.enable_gradient_checkpointing()

    assert backbone.transformer.gradient_checkpointing is True
    assert backbone.vision_backbone.gradient_checkpointing is True
    assert backbone.action_expert is not None
    assert backbone.action_expert.gradient_checkpointing is True

    model.freeze_vlm()
    assert all(parameter.requires_grad for parameter in backbone.action_expert.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.transformer.parameters())


def test_enable_compile_wraps_public_entrypoints(
    model: MolmoAct2Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[str] = []

    def compile_method(method: object, *, mode: str) -> object:
        assert mode == "default"
        compiled.append(method.__name__)  # type: ignore[attr-defined]
        return method

    monkeypatch.setattr(torch, "compile", compile_method)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda precision: None)

    model.enable_compile()

    assert compiled == ["forward", "predict_action_chunk"]


def test_enable_lora_creates_trainable_adapters(model: MolmoAct2Model) -> None:
    pytest.importorskip("peft")

    model.enable_lora()

    assert any("lora_" in name and parameter.requires_grad for name, parameter in model.named_parameters())
