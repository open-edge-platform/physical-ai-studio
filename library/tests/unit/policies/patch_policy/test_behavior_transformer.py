# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real residual tokenizer and Behavior Transformer tests on CPU."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.policies.patch_policy.components.heads.behavior_transformer import (
    BehaviourTransformerHead,
    ResidualActionQuantizer,
)
from physicalai.policies.patch_policy.model import PatchPolicyModel


def _make_action_quantizer() -> ResidualActionQuantizer:
    torch.manual_seed(13)
    return ResidualActionQuantizer(1, 2, latent_dim=4, num_embeddings=2, groups=2, action_scale=2.0)


def _make_small_behavior_head(*, groups: int = 2, action_dim: int = 1) -> BehaviourTransformerHead:
    torch.manual_seed(17)
    return BehaviourTransformerHead(
        context_dim=2,
        action_dim=action_dim,
        chunk_size=2,
        n_obs_steps=2,
        n_patches=1,
        hidden_dim=4,
        num_layers=1,
        num_heads=1,
        latent_dim=4,
        num_embeddings=2,
        groups=groups,
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
    assert loss.shape == (1, 2, 2, 1)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
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


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_unreduced_loss_mean_matches_upstream_objective(groups: int) -> None:
    """Window losses preserve group weighting and the upstream scalar mean."""
    head = _make_small_behavior_head(groups=groups, action_dim=2).eval()
    head.offset_loss_weight = 3.0
    head.secondary_code_weight = 0.7
    actions = torch.tensor([[[0.1, 0.2], [-0.2, 0.5], [0.3, -0.4]], [[0.4, 0.1], [0.6, -0.3], [0.8, 0.9]]])
    context = {"tokens": _behavior_context()["tokens"].expand(2, -1, -1)}
    windows = actions.unfold(1, head.chunk_size, 1).transpose(-1, -2)
    logits, offsets = head._predictions(context)
    codes = head.quantizer.encode(windows.flatten(0, 1)).reshape(2, 2, groups)
    torch.manual_seed(29)
    predicted = head._actions(logits, offsets)
    # Upstream computes one mean focal loss per group, then sums weighted groups.
    group_losses = []
    for group in range(groups):
        logpt = F.log_softmax(logits[:, :, group].flatten(0, 1), dim=-1)
        logpt = logpt.gather(1, codes[:, :, group].reshape(-1, 1)).view(-1)
        group_losses.append((-((1 - logpt.exp()) ** head.focal_loss_gamma) * logpt).mean())
    expected = 5 * group_losses[0] + head.secondary_code_weight * sum(group_losses[1:])
    expected += head.offset_loss_weight * F.l1_loss(predicted, windows)

    torch.manual_seed(29)
    losses = head.compute_loss(actions, context)

    assert losses.shape == (2, 2, 2, 2)
    torch.testing.assert_close(losses.mean(), expected)
    logpt = logits.log_softmax(-1).gather(-1, codes[..., None]).squeeze(-1)
    focal = -((1 - logpt.exp()) ** head.focal_loss_gamma) * logpt
    window_code_loss = 5 * focal[..., 0] + head.secondary_code_weight * focal[..., 1:].sum(-1)
    torch.testing.assert_close(
        losses,
        window_code_loss[..., None, None] + head.offset_loss_weight * (predicted - windows).abs(),
    )


def test_extreme_logits_have_finite_nonzero_focal_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrong codes retain upstream focal gradients even after softmax underflows."""
    head = _make_small_behavior_head().eval()
    head.offset_loss_weight = 0.0
    actions = torch.tensor([[[0.1], [-0.2], [0.3]]])
    windows = actions.unfold(1, 2, 1).transpose(-1, -2)
    codes = head.quantizer.encode(windows.flatten(0, 1)).reshape(1, 2, 2)
    logits = torch.full((1, 2, 2, 2), 1000.0).scatter_(-1, codes[..., None], -1000.0).requires_grad_()
    offsets = torch.zeros(1, 2, 2, 2, 2, 1)
    monkeypatch.setattr(head, "_predictions", lambda _context: (logits, offsets))

    loss = head.compute_loss(actions, _behavior_context()).mean()
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(11000.0))
    assert torch.isfinite(logits.grad).all()
    target_gradients = logits.grad.gather(-1, codes[..., None]).squeeze(-1)
    torch.testing.assert_close(target_gradients, torch.tensor([[[-2.5, -0.25], [-2.5, -0.25]]]))


def test_loss_rejects_mask_argument_instead_of_silently_ignoring_it() -> None:
    """The head follows the shared mask-free compute_loss signature."""
    head = _make_small_behavior_head().eval()
    with pytest.raises(TypeError, match="valid_steps"):
        head.compute_loss(torch.zeros(1, 3, 1), _behavior_context(), valid_steps=torch.ones(1, 3))


def test_model_reduces_all_behavior_windows_including_padding(
    tiny_encoder,
    small_config,
    model_batch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trajectory padding does not mask or misalign overlapping VQ-BeT windows."""
    model = PatchPolicyModel.from_config(replace(small_config, action_head="behavior_transformer"))
    losses = torch.arange(8.0).reshape(1, 2, 2, 2).requires_grad_()
    monkeypatch.setattr(model.action_head, "compute_loss", lambda _actions, _context: losses)
    model_batch["extra.action_is_pad"] = torch.tensor([[False, False, True]])

    loss, metrics = model.compute_loss(model_batch)
    loss.backward()

    torch.testing.assert_close(loss, losses.mean())
    torch.testing.assert_close(losses.grad, torch.full_like(losses, 1 / losses.numel()))
    assert not metrics["loss"].requires_grad


def test_argmax_sampling_is_repeatable_without_random_ops(monkeypatch) -> None:
    head = _make_small_behavior_head().eval()
    head.use_random_noise = False
    context = _behavior_context()

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("Argmax inference must not sample random codes")

    monkeypatch.setattr(torch, "multinomial", forbid_sampling)
    with torch.no_grad():
        logits, offsets = head._predictions(context)
        logits = torch.tensor([[[[0.0, 2.0], [3.0, 0.0]], [[0.0, 4.0], [0.0, 5.0]]]])
        monkeypatch.setattr(head, "_predictions", lambda context: (logits, offsets))
        codes = torch.tensor([[[1, 0], [1, 1]]])
        expected = head._actions(logits, offsets, codes)[:, -1]
        rng_state = torch.get_rng_state()
        first = head.sample(context)
        second = head.sample(context)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, first)
    assert torch.equal(torch.get_rng_state(), rng_state)
