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


def test_action_expert_context_metadata_masks_padded_horizon(model: MolmoAct2Model) -> None:
    action_expert = model.backbone.model.action_expert
    assert action_expert is not None
    horizon_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])

    cross_mask, self_mask, valid_action, _ = action_expert.prepare_context_metadata(
        encoder_attention_mask=torch.ones(2, 3, dtype=torch.bool),
        seq_len=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        action_horizon_is_pad=horizon_mask,
    )

    assert cross_mask is not None
    assert self_mask is not None
    assert valid_action is not None
    expected_self_mask = horizon_mask[:, None, None, :].float() * torch.finfo(torch.float32).min
    torch.testing.assert_close(self_mask, expected_self_mask)
    torch.testing.assert_close(valid_action, (~horizon_mask).float().unsqueeze(-1))


def test_predict_flow_velocity_forwards_horizon_mask(
    model: MolmoAct2Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = model.backbone.model
    horizon_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    captured: dict[str, object] = {}

    def predict_per_layer(**kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.zeros_like(kwargs["x_t"])

    monkeypatch.setattr(backbone, "_predict_flow_velocity_per_layer", predict_per_layer)
    predicted, target = backbone.predict_flow_velocity(
        input_ids=torch.zeros(2, 1, dtype=torch.long),
        attention_mask=torch.ones(2, 1, dtype=torch.bool),
        token_type_ids=None,
        images=None,
        token_pooling=None,
        actions=torch.zeros(2, 4, 4),
        action_horizon_is_pad=horizon_mask,
        action_dim_is_pad=None,
        freeze_encoder=False,
    )

    assert captured["action_horizon_is_pad"] is horizon_mask
    assert predicted.shape == target.shape == (2, backbone.num_flow_timesteps, 4, 4)


def test_flow_prediction_ignores_padded_action_tail(
    model: MolmoAct2Model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = model.backbone.model
    backbone.num_flow_timesteps = 1
    horizon_mask = torch.tensor([[False, False, True, True]])

    def deterministic_interpolation(
        actions: torch.Tensor,
        _action_dim_is_pad: torch.Tensor | None,
        _dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return actions, torch.zeros(actions.shape[0]), torch.zeros_like(actions)

    monkeypatch.setattr(backbone, "_flow_interpolation", deterministic_interpolation)
    inputs = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "token_type_ids": None,
        "images": None,
        "token_pooling": None,
        "action_horizon_is_pad": horizon_mask,
        "action_dim_is_pad": None,
        "freeze_encoder": False,
    }
    actions = torch.zeros(1, 4, 4)
    padded_tail_changed = actions.clone()
    padded_tail_changed[:, 2:] = 100.0

    predicted, _ = backbone.predict_flow_velocity(actions=actions, **inputs)
    changed, _ = backbone.predict_flow_velocity(actions=padded_tail_changed, **inputs)

    torch.testing.assert_close(predicted[:, :, :2], changed[:, :, :2])
    torch.testing.assert_close(predicted[:, :, 2:], torch.zeros_like(predicted[:, :, 2:]))
    torch.testing.assert_close(changed[:, :, 2:], torch.zeros_like(changed[:, :, 2:]))


def test_forward_dispatches_by_mode(model: MolmoAct2Model, monkeypatch: pytest.MonkeyPatch) -> None:
    loss = torch.tensor(1.0)
    actions = torch.ones(1, 2, 4)
    monkeypatch.setattr(model, "compute_loss", lambda _: (loss, {"loss": loss}))
    monkeypatch.setattr(model, "predict_action_chunk", lambda _: actions)

    model.train()
    assert model({})[0] is loss
    model.eval()
    assert model({}) is actions


def test_predict_action_chunk_returns_full_chunk(
    model: MolmoAct2Model,
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = torch.ones(1, tiny_molmoact2_config.chunk_size, tiny_molmoact2_config.max_action_dim)
    calls: dict[str, object] = {}

    def generate(**kwargs: object) -> torch.Tensor:
        calls.update(kwargs)
        return generated

    monkeypatch.setattr(model.backbone.model, "generate_actions_from_inputs", generate)
    actions = model.predict_action_chunk({"input_ids": torch.zeros(1, 1, dtype=torch.long)})

    assert actions.shape == (1, tiny_molmoact2_config.chunk_size, 4)
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
    backbone = model.backbone.model
    model.enable_gradient_checkpointing()

    assert backbone.transformer.gradient_checkpointing is True
    assert backbone.vision_backbone.gradient_checkpointing is True
    assert backbone.action_expert is not None
    assert backbone.action_expert.gradient_checkpointing is True

    model.freeze_vlm()
    assert all(parameter.requires_grad for parameter in backbone.action_expert.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.transformer.parameters())


def test_enable_compile_wraps_inference_entrypoint(
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

    assert compiled == ["predict_action_chunk"]


def test_default_peft_targets_include_vlm_only() -> None:
    targets = MolmoAct2Model.get_default_peft_targets()

    assert "vision_backbone" in targets
    assert "action_expert" not in targets
