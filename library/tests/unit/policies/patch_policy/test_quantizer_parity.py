# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101

"""Portable checks of the vendored upstream VQ-VAE's default training equations."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from physicalai.policies.patch_policy.components.heads.behavior_transformer import ResidualActionQuantizer


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> str:
    """Exercise CPU and available CUDA without requiring accelerator hardware.

    Returns:
        The requested available device type.
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_training_matches_upstream_equations_without_dead_replacement(device: str, groups: int) -> None:  # noqa: PLR0914
    """Compare loss, gradients, EMA buffers, and an optimizer step with dense upstream equations."""
    with torch.random.fork_rng():
        torch.manual_seed(321)
        quantizer = ResidualActionQuantizer(2, 3, 8, 4, groups, action_scale=2.5).to(device)
        actions = torch.randn(11, 3, 2, device=device)
    encoder = deepcopy(quantizer.encoder)
    decoder = deepcopy(quantizer.decoder)
    codebook = quantizer.codebook.clone()
    counts = torch.full_like(quantizer.cluster_size, 100)
    averages = codebook * counts[..., None]
    quantizer.cluster_size.copy_(counts)
    quantizer.embed_avg.copy_(averages)

    target = (actions / 2.5).flatten(-2)
    residual = encoder(target)
    quantized = torch.zeros_like(residual)
    commitment = residual.new_zeros(())
    expected_codes = []
    for group in range(groups):
        # Upstream computes expanded squared distances, then takes sqrt and argmin.
        distances = (
            residual.square().sum(-1, keepdim=True)
            + codebook[group].square().sum(-1)[None]
            - 2 * residual @ codebook[group].T
        ).sqrt()
        codes = distances.argmin(-1)
        expected_codes.append(codes)
        onehot = F.one_hot(codes, codebook.shape[1]).to(residual)
        selected = onehot @ codebook[group]
        commitment = torch.add(commitment, F.mse_loss(selected.detach(), residual))
        stage = residual + (selected - residual).detach()
        counts[group].lerp_(onehot.sum(0), 0.2)
        averages[group].lerp_(onehot.T @ residual.detach(), 0.2)
        total = counts[group].sum()
        smoothed = (counts[group] + 1e-5) / (total + codebook.shape[1] * 1e-5) * total
        codebook[group] = averages[group] / smoothed[:, None]
        residual = torch.sub(residual, stage.detach())
        quantized = torch.add(quantized, stage)

    decoded = decoder(quantized)
    reconstruction = F.l1_loss(decoded, target)
    expected_loss = reconstruction + 5 * commitment
    torch.testing.assert_close(quantizer.encode(actions), torch.stack(expected_codes, -1))
    loss, metrics = quantizer.compute_loss(actions)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(metrics["encoder_loss"], reconstruction)
    torch.testing.assert_close(metrics["vq_loss_state"], commitment)
    torch.testing.assert_close(metrics["vqvae_recon_loss"], F.mse_loss(decoded, target))
    torch.testing.assert_close(quantizer.codebook, codebook)
    torch.testing.assert_close(quantizer.cluster_size, counts)
    torch.testing.assert_close(quantizer.embed_avg, averages)

    loss.backward()
    expected_loss.backward()
    for actual_module, expected_module in [(quantizer.encoder, encoder), (quantizer.decoder, decoder)]:
        for actual, expected in zip(actual_module.parameters(), expected_module.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)
    torch.optim.SGD(quantizer.parameters(), lr=0.01).step()
    torch.optim.SGD([*encoder.parameters(), *decoder.parameters()], lr=0.01).step()
    for actual_module, expected_module in [(quantizer.encoder, encoder), (quantizer.decoder, decoder)]:
        for actual, expected in zip(actual_module.parameters(), expected_module.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("batch_size", [2, 7])
def test_dead_replacement_matches_upstream_device_sampling(device: str, batch_size: int) -> None:
    """Check replacement values and RNG consumption on the residual's device."""
    num_expired = 4
    quantizer = ResidualActionQuantizer(1, 1, 3, 4, 1, action_scale=1.0).to(device)
    residual = torch.arange(batch_size * 3, device=device, dtype=torch.float32).reshape(batch_size, 3)
    codes = torch.zeros(batch_size, device=device, dtype=torch.long)
    # All four codes expire after EMA. Exercise both sampling branches with distinct vectors.
    with torch.random.fork_rng():
        torch.manual_seed(123)
        if batch_size >= num_expired:
            indices = torch.randperm(batch_size, device=device)[:num_expired]
        else:
            indices = torch.randint(0, batch_size, (num_expired,), device=device)
        expected = residual[indices]
        expected_cpu_rng = torch.get_rng_state()
        expected_device_rng = torch.cuda.get_rng_state() if device == "cuda" else expected_cpu_rng
        torch.manual_seed(123)
        quantizer._update_codebook(0, residual, codes)  # noqa: SLF001 - isolate replacement sampling
        torch.testing.assert_close(quantizer.codebook[0], expected)
        torch.testing.assert_close(quantizer.cluster_size[0], torch.full((4,), 2.0, device=device))
        torch.testing.assert_close(quantizer.embed_avg[0], expected * 2)
        assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
        actual_device_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        assert torch.equal(actual_device_rng, expected_device_rng)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_stacked_initialization_has_upstream_per_codebook_fan_in(groups: int) -> None:
    """Keep the upstream three-dimensional fan-in, including the codebook-size factor."""
    with torch.random.fork_rng():
        torch.manual_seed(42)
        stacked = nn.init.kaiming_uniform_(torch.empty(groups, 16, 512))
        torch.manual_seed(42)
        upstream = torch.cat([nn.init.kaiming_uniform_(torch.empty(1, 16, 512)) for _ in range(groups)])
    torch.testing.assert_close(stacked, upstream, atol=0, rtol=0)
    assert stacked.abs().max() <= (6 / (16 * 512)) ** 0.5


def test_eval_matches_upstream_zero_commitment() -> None:
    """Eval uses reconstruction only and does not backpropagate through the encoder."""
    quantizer = ResidualActionQuantizer(1, 1, 1, 2, 2, action_scale=1.0).eval()
    quantizer.encoder = nn.Linear(1, 1, bias=False)
    quantizer.decoder = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        quantizer.encoder.weight.fill_(1)
        quantizer.decoder.weight.fill_(1)
        quantizer.codebook.copy_(torch.tensor([[[1.0], [10.0]], [[0.5], [10.0]]]))
    loss, metrics = quantizer.compute_loss(torch.tensor([[[2.0]]]))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.5))
    torch.testing.assert_close(metrics["vq_loss_state"], torch.tensor(0.0))
    assert quantizer.encoder.weight.grad is None
    torch.testing.assert_close(quantizer.decoder.weight.grad, torch.tensor([[-1.5]]))


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_seeded_initialization_matches_upstream_order(groups: int) -> None:
    """Initialize each codebook before either MLP, matching upstream RNG consumption."""
    with torch.random.fork_rng():
        torch.manual_seed(42)
        quantizer = ResidualActionQuantizer(2, 3, 8, 4, groups, action_scale=1.0)
        actual_rng = torch.get_rng_state()
        torch.manual_seed(42)
        codebook = torch.cat([nn.init.kaiming_uniform_(torch.empty(1, 4, 8)) for _ in range(groups)])
        encoder = type(quantizer.encoder)(6, 8)
        decoder = type(quantizer.decoder)(8, 6)
        torch.testing.assert_close(quantizer.codebook, codebook, atol=0, rtol=0)
        torch.testing.assert_close(quantizer.embed_avg, codebook, atol=0, rtol=0)
        for actual_module, expected_module in [(quantizer.encoder, encoder), (quantizer.decoder, decoder)]:
            for actual, expected in zip(actual_module.parameters(), expected_module.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(torch.get_rng_state(), actual_rng)
