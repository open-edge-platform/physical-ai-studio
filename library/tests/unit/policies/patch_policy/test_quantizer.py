# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for action tokenizer fitting and frozen head integration."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from physicalai.policies.patch_policy.components.heads.behavior_transformer import (
    BehaviourTransformerHead,
    ResidualActionQuantizer,
)


def scalar_quantizer(*, groups: int = 2, action_scale: float = 1.0) -> ResidualActionQuantizer:
    quantizer = ResidualActionQuantizer(1, 1, 1, 2, groups, action_scale=action_scale)
    quantizer.encoder = nn.Linear(1, 1, bias=False)
    quantizer.decoder = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        quantizer.encoder.weight.fill_(1)
        quantizer.decoder.weight.fill_(1)
        quantizer.codebook.copy_(torch.tensor([[[1.0], [10.0]], [[0.5], [10.0]]])[:groups])
        quantizer.embed_avg.copy_(quantizer.codebook)
    return quantizer


def small_head() -> BehaviourTransformerHead:
    return BehaviourTransformerHead(
        4,
        1,
        2,
        n_obs_steps=2,
        n_patches=1,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        latent_dim=2,
        num_embeddings=2,
        groups=2,
    )


def test_training_uses_preupdate_targets_and_per_stage_straight_through() -> None:
    quantizer = scalar_quantizer()
    actions = torch.tensor([[[2.0]]])

    loss, metrics = quantizer.compute_loss(actions)
    loss.backward()

    # Stage residuals are 2 and 1; old codes are 1 and .5, even after EMA replaces them.
    torch.testing.assert_close(metrics["encoder_loss"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["vqvae_recon_loss"], torch.tensor(0.25))
    torch.testing.assert_close(metrics["vq_loss_state"], torch.tensor(1.25))
    torch.testing.assert_close(loss, torch.tensor(6.75))
    # Reconstruction contributes -2 (one ST path per stage), commitment contributes 15.
    torch.testing.assert_close(quantizer.encoder.weight.grad, torch.tensor([[26.0]]))
    torch.testing.assert_close(quantizer.decoder.weight.grad, torch.tensor([[-1.5]]))
    assert not quantizer.fitted
    assert all(not value.requires_grad for value in metrics.values())
    assert quantizer.codebook.grad is None
    assert not torch.equal(quantizer.codebook, torch.tensor([[[1.0], [10.0]], [[0.5], [10.0]]]))


def test_action_scaling_and_residual_encoding_decoding() -> None:
    quantizer = scalar_quantizer(action_scale=2.0).eval()
    actions = torch.tensor([[[4.0]], [[23.0]]])

    codes = quantizer.encode(actions)
    decoded = quantizer.decode_codes(codes)
    loss, metrics = quantizer.compute_loss(actions[:1])

    torch.testing.assert_close(codes, torch.tensor([[0, 0], [1, 0]]))
    torch.testing.assert_close(decoded, torch.tensor([[[3.0]], [[21.0]]]))
    torch.testing.assert_close(loss, torch.tensor(0.5))
    torch.testing.assert_close(metrics["vq_loss_state"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["encoder_loss"], torch.tensor(0.5))


def test_ema_decay_and_laplace_smoothing() -> None:
    quantizer = scalar_quantizer(groups=1)
    with torch.no_grad():
        quantizer.cluster_size.copy_(torch.tensor([[5.0, 10.0]]))
        quantizer.embed_avg.copy_(quantizer.codebook * quantizer.cluster_size[..., None])
    actions = torch.tensor([[[0.9]], [[1.1]], [[9.0]], [[11.0]]])
    old_average = quantizer.embed_avg.clone()

    quantizer.compute_loss(actions)

    expected_counts = torch.tensor([[4.4, 8.4]])
    expected_average = old_average * 0.8 + torch.tensor([[[2.0], [20.0]]]) * 0.2
    smoothed = (expected_counts + 1e-5) / (expected_counts.sum(-1, keepdim=True) + 2e-5)
    smoothed *= expected_counts.sum(-1, keepdim=True)
    torch.testing.assert_close(quantizer.cluster_size, expected_counts)
    torch.testing.assert_close(quantizer.embed_avg, expected_average)
    torch.testing.assert_close(quantizer.codebook, expected_average / smoothed[..., None])


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dead_codes_are_replaced_from_stage_residuals(batch_size: int) -> None:
    quantizer = scalar_quantizer()
    quantizer.compute_loss(torch.full((batch_size, 1, 1), 2.0))

    torch.testing.assert_close(quantizer.cluster_size, torch.full((2, 2), 2.0))
    torch.testing.assert_close(quantizer.codebook, torch.tensor([[[2.0], [2.0]], [[1.0], [1.0]]]))
    torch.testing.assert_close(quantizer.embed_avg, quantizer.codebook * 2)


@pytest.mark.parametrize("training", [True, False])
def test_encode_and_decode_never_mutate_or_build_gradients(training: bool) -> None:
    quantizer = scalar_quantizer().train(training)
    before = deepcopy(quantizer.state_dict())
    actions = torch.tensor([[[2.0]]], requires_grad=True)

    codes = quantizer.encode(actions)
    decoded = quantizer.decode_codes(codes)

    assert not decoded.requires_grad
    for key, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("training,trainable", [(False, True), (True, False)])
def test_loss_does_not_update_ema_when_eval_or_frozen(training: bool, trainable: bool) -> None:
    quantizer = scalar_quantizer().requires_grad_(trainable).train(training)
    before = deepcopy(quantizer.state_dict())
    quantizer.compute_loss(torch.tensor([[[2.0]]]))
    for key, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_frozen_head_supports_explicit_fit_copy_and_state_transfer() -> None:
    head = small_head()
    assert not head.quantizer.training
    assert not any(parameter.requires_grad for parameter in head.quantizer.parameters())
    before = deepcopy(head.quantizer.state_dict())
    fitting = deepcopy(head.quantizer).requires_grad_(True).train()
    optimizer = torch.optim.Adam(fitting.parameters(), lr=1e-3)
    actions = torch.randn(4, 2, 1)
    loss, _ = fitting.compute_loss(actions)
    loss.backward()
    optimizer.step()

    assert any(parameter.grad is not None for parameter in fitting.encoder.parameters())
    assert any(parameter.grad is not None for parameter in fitting.decoder.parameters())
    assert not fitting.fitted
    for key, value in head.quantizer.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert not torch.equal(fitting.codebook, head.quantizer.codebook)
    fitting.fitted.fill_(True)
    head.quantizer.load_state_dict(fitting.state_dict())
    for mode in (True, False, True):
        assert head.train(mode) is head
        assert not head.quantizer.training
        assert not any(parameter.requires_grad for parameter in head.quantizer.parameters())

    restored = small_head()
    restored.load_state_dict(head.state_dict())
    assert restored.quantizer.fitted.dtype == torch.bool
    assert restored.quantizer.fitted
    assert "quantizer.codebook" in head.state_dict()
    assert "quantizer.encoder.encoder.0.weight" in head.state_dict()
    assert "quantizer.decoder.fc.weight" in head.state_dict()
    assert "codebook" not in dict(fitting.named_parameters())


def test_head_training_requires_fitted_tokenizer_and_keeps_it_frozen() -> None:
    head = small_head()
    actions = torch.randn(2, 3, 1)
    context = {"tokens": torch.randn(2, 2, 4)}
    with pytest.raises(RuntimeError, match="Fit the action quantizer"):
        head.compute_loss(actions, context)

    head.quantizer.fitted.fill_(True)
    before = deepcopy(head.quantizer.state_dict())
    loss = head.compute_loss(actions, context)
    loss.backward()
    assert any(parameter.grad is not None for parameter in head.trunk.parameters())
    assert any(parameter.grad is not None for parameter in head.code_head.parameters())
    assert any(parameter.grad is not None for parameter in head.offset_head.parameters())
    assert all(parameter.grad is None for parameter in head.quantizer.parameters())
    for key, value in head.quantizer.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_unfitted_head_eval_loss_and_sampling_remain_available() -> None:
    head = small_head().eval()
    context = {"tokens": torch.randn(2, 2, 4)}
    assert torch.isfinite(head.compute_loss(torch.randn(2, 3, 1), context))
    assert head.sample(context).shape == (2, 2, 1)


@pytest.mark.parametrize("shape", [(0, 1, 1), (2, 1), (2, 2, 1), (2, 1, 2)])
def test_fit_rejects_empty_or_incompatible_actions(shape: tuple[int, ...]) -> None:
    quantizer = scalar_quantizer()
    with pytest.raises(ValueError, match="Expected nonempty action windows"):
        quantizer.compute_loss(torch.zeros(shape))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_quantizer_rejects_invalid_action_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="action_scale must be finite and positive"):
        scalar_quantizer(action_scale=scale)


@pytest.mark.parametrize("dimension", [0, -1])
def test_quantizer_rejects_invalid_dimensions(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimensions and groups must be positive"):
        ResidualActionQuantizer(1, 1, 1, 2, dimension, action_scale=1.0)
