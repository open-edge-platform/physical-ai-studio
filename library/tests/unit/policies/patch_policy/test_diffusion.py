# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real diffusion masks, gradients, and current-chunk sampling on CPU."""

import pytest
import torch

from physicalai.policies.components import DiffusionActionHead
from physicalai.policies.components.action_heads.diffusion.diffusion import PredictionType
from physicalai.policies.patch_policy.components.heads.diffusion import DiffusionHead


def _make_small_diffusion(prediction_type: PredictionType = "epsilon") -> DiffusionHead:
    torch.manual_seed(11)
    return DiffusionHead(
        context_dim=2,
        action_dim=1,
        chunk_size=2,
        n_obs_steps=2,
        n_patches=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        num_train_timesteps=4,
        num_inference_steps=2,
        prediction_type=prediction_type,
        eta=0.0,
        use_random_input_noise=False,
    )


def _diffusion_context() -> dict[str, torch.Tensor]:
    return {"tokens": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])}


def test_target_mask_is_action_causal() -> None:
    head = _make_small_diffusion()
    blocked = float("-inf")
    expected = torch.tensor([[0.0, blocked, blocked], [0.0, 0.0, blocked], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(head.target_mask, expected)
    assert "target_mask" not in head.state_dict()


def test_memory_mask_includes_time_and_only_observed_patches() -> None:
    head = _make_small_diffusion()
    blocked = float("-inf")
    expected = torch.tensor([[0.0, 0.0, 0.0, blocked, blocked], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(head.memory_mask, expected)
    assert "memory_mask" not in head.state_dict()


def test_denoise_cannot_see_future_actions_or_observations() -> None:
    head = _make_small_diffusion().eval()
    actions = torch.tensor([[[0.1], [0.2], [0.3]]])
    context = _diffusion_context()
    changed_actions = torch.tensor([[[0.1], [-4.0], [5.0]]])
    changed_tokens = context["tokens"].clone()
    changed_tokens[:, 2:] = torch.tensor([[[-2.0, 7.0], [6.0, -3.0]]])
    output = head.denoise(actions, torch.tensor([2]), context)
    changed = head.denoise(changed_actions, torch.tensor([2]), {"tokens": changed_tokens})
    assert output.shape == actions.shape
    torch.testing.assert_close(output[:, 0], changed[:, 0])
    assert not torch.allclose(output[:, 1:], changed[:, 1:])


def test_bfloat16_denoise_outside_autocast_preserves_time_precision() -> None:
    """FP32 time features enter the BF16 network without promoting its memory."""
    head = _make_small_diffusion().to(torch.bfloat16)
    actions = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.bfloat16)
    context = {"tokens": _diffusion_context()["tokens"].to(torch.bfloat16).requires_grad_()}
    timesteps = torch.tensor([999])
    with torch.autocast(device_type="cpu", enabled=False):
        embedding = head.time_encoder(timesteps)
        output = head.denoise(actions, timesteps, context)
        output.float().square().mean().backward()

    frequencies = torch.exp(torch.arange(4, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 3))
    values = timesteps.float()[:, None] * frequencies
    assert embedding.dtype == torch.float32
    torch.testing.assert_close(embedding, torch.cat((values.sin(), values.cos()), dim=-1))
    assert output.dtype == torch.bfloat16
    assert output.shape == actions.shape
    assert torch.isfinite(output).all()
    for gradient in (context["tokens"].grad, head.context_in.weight.grad, head.action_out.weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_loss_is_unreduced_and_reaches_context_and_head(prediction_type: PredictionType) -> None:
    head = _make_small_diffusion(prediction_type)
    context = _diffusion_context()
    context["tokens"].requires_grad_()
    actions = torch.tensor([[[0.1], [-0.2], [0.3]]])
    loss = head.compute_loss(actions, context)
    assert loss.shape == actions.shape
    assert torch.isfinite(loss).all()
    assert (loss >= 0).all()
    loss.mean().backward()
    for gradient in (context["tokens"].grad, head.context_in.weight.grad, head.action_out.weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_sample_returns_current_horizon_from_full_trajectory() -> None:
    head = _make_small_diffusion().eval()
    context = _diffusion_context()
    noise = torch.tensor([[[0.1], [-0.2], [0.3]]])
    with torch.no_grad():
        full = DiffusionActionHead.sample(head, context, noise=noise, num_steps=2)
        chunk = head.sample(context, noise=noise, num_steps=2)
    assert head.chunk_size == 3
    assert full.shape == (1, 3, 1)
    assert chunk.shape == (1, 2, 1)
    assert torch.isfinite(chunk).all()
    torch.testing.assert_close(chunk, full[:, 1:3])
    torch.testing.assert_close(noise, torch.tensor([[[0.1], [-0.2], [0.3]]]))


def test_zero_input_sampling_is_repeatable() -> None:
    head = _make_small_diffusion().eval()
    with torch.no_grad():
        first = head.sample(_diffusion_context())
        second = head.sample(_diffusion_context())
    torch.testing.assert_close(first, second)
    assert first.shape == (1, 2, 1)
