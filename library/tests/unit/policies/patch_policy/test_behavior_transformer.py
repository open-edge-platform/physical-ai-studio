# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real residual tokenizer and Behavior Transformer tests on CPU."""

import pytest
import torch

from physicalai.policies.patch_policy.components.heads.behavior_transformer import (
    BehaviourTransformerHead,
    ResidualActionQuantizer,
)


def _make_action_quantizer() -> ResidualActionQuantizer:
    torch.manual_seed(13)
    return ResidualActionQuantizer(1, 2, latent_dim=4, num_embeddings=2, groups=2, action_scale=2.0)


def _make_small_behavior_head() -> BehaviourTransformerHead:
    torch.manual_seed(17)
    return BehaviourTransformerHead(
        context_dim=2,
        action_dim=1,
        chunk_size=2,
        n_obs_steps=2,
        n_patches=1,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        latent_dim=4,
        num_embeddings=2,
        groups=2,
        action_scale=2.0,
    )


def _behavior_context() -> dict[str, torch.Tensor]:
    return {"tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}


def test_quantizer_training_reaches_encoder_decoder_and_updates_ema() -> None:
    quantizer = _make_action_quantizer().train()
    actions = torch.tensor([[[1.0], [-2.0]], [[3.0], [4.0]]])
    before = quantizer.cluster_size.clone()
    loss, metrics = quantizer.compute_loss(actions)
    loss.backward()
    for gradient in (quantizer.encoder.fc.weight.grad, quantizer.decoder.fc.weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert not torch.equal(quantizer.cluster_size, before)
    assert not quantizer.codebook.requires_grad
    assert not quantizer.fitted
    torch.testing.assert_close(loss.detach(), metrics["encoder_loss"] + 5 * metrics["vq_loss_state"])
    assert all(not metric.requires_grad for metric in metrics.values())


def test_quantizer_eval_keeps_state_and_has_no_encoder_gradient() -> None:
    quantizer = _make_action_quantizer().eval()
    actions = torch.tensor([[[1.0], [-2.0]], [[3.0], [4.0]]])
    before = {name: value.clone() for name, value in quantizer.state_dict().items()}
    loss, metrics = quantizer.compute_loss(actions)
    loss.backward()
    assert all(parameter.grad is None for parameter in quantizer.encoder.parameters())
    assert quantizer.decoder.fc.weight.grad.abs().sum() > 0
    torch.testing.assert_close(metrics["vq_loss_state"], torch.tensor(0.0))
    torch.testing.assert_close(loss.detach(), metrics["encoder_loss"])
    for name, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_quantizer_encode_decode_preserve_state_and_window_shape() -> None:
    quantizer = _make_action_quantizer().train()
    actions = torch.tensor([[[1.0], [-2.0]], [[3.0], [4.0]]])
    before = {name: value.clone() for name, value in quantizer.state_dict().items()}
    codes = quantizer.encode(actions)
    decoded = quantizer.decode_codes(codes)
    assert codes.shape == (2, 2)
    assert codes.dtype == torch.long
    assert ((codes >= 0) & (codes < 2)).all()
    assert decoded.shape == (2, 2, 1)
    assert not decoded.requires_grad
    assert torch.isfinite(decoded).all()
    for name, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_head_training_keeps_quantizer_frozen_and_requires_fitting() -> None:
    head = _make_small_behavior_head()
    assert head.train() is head
    assert head.training
    assert not head.quantizer.training
    assert all(not parameter.requires_grad for parameter in head.quantizer.parameters())
    actions = torch.tensor([[[0.1], [-0.2], [0.3]]])
    with pytest.raises(RuntimeError, match="Fit the action quantizer"):
        head.compute_loss(actions, _behavior_context())
    head.quantizer.fitted.fill_(True)
    before = {name: value.clone() for name, value in head.quantizer.state_dict().items()}
    loss = head.compute_loss(actions, _behavior_context())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    for gradient in (head.trunk.lm_head.weight.grad, head.code_head[0].weight.grad, head.offset_head[0].weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert all(parameter.grad is None for parameter in head.quantizer.parameters())
    for name, value in head.quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_actions_sum_selected_group_offsets_and_real_quantized_centers() -> None:
    head = _make_small_behavior_head().eval()
    logits = torch.zeros(1, 2, 2, 2)
    codes = torch.tensor([[[0, 1], [1, 0]]])
    offsets = (
        torch
        .tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        )
        .reshape(1, 2, 2, 2, 2, 1)
        .requires_grad_()
    )
    centers = head.quantizer.decode_codes(codes.reshape(2, 2)).reshape(1, 2, 2, 1)
    actions = head._actions(logits, offsets, codes)
    expected_offsets = torch.tensor([[[[8.0], [10.0]], [[24.0], [26.0]]]])
    torch.testing.assert_close(actions, centers + expected_offsets)
    actions.sum().backward()
    assert offsets.grad.sum() == 8
    torch.testing.assert_close(offsets.grad[0, 0, 0, 1], torch.zeros(2, 1))


def test_sample_selects_final_observation_chunk_without_mutating_quantizer() -> None:
    head = _make_small_behavior_head().eval()
    context = _behavior_context()
    before = {name: value.clone() for name, value in head.quantizer.state_dict().items()}
    with torch.no_grad():
        logits, offsets = head._predictions(context)
        torch.manual_seed(19)
        expected = head._actions(logits, offsets)[:, -1]
        torch.manual_seed(19)
        sampled = head.sample(context)
    assert sampled.shape == (1, 2, 1)
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled, expected)
    for name, value in head.quantizer.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_loss_rejects_short_action_trajectory() -> None:
    head = _make_small_behavior_head().eval()
    with pytest.raises(ValueError, match="Expected 3 action steps, got 2"):
        head.compute_loss(torch.tensor([[[0.1], [0.2]]]), _behavior_context())
