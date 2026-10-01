# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the action head base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from physicalai.policies.components import ActionHead, IterativeActionHead
from physicalai.policies.components.action_heads import Context

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

CHUNK_SIZE, ACTION_DIM, CONTEXT_DIM = 4, 3, 8


class ToyFlowHead(IterativeActionHead):
    """Minimal Euler flow-matching head used to exercise the base class."""

    def __init__(self) -> None:
        super().__init__(CHUNK_SIZE, ACTION_DIM, num_inference_steps=5)
        self.velocity = nn.Linear(ACTION_DIM + CONTEXT_DIM + 1, ACTION_DIM)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        pooled = context["tokens"].mean(dim=1, keepdim=True).expand(-1, x_t.shape[1], -1)
        time = t[:, None, None].expand(-1, x_t.shape[1], 1)
        return self.velocity(torch.cat([x_t, pooled, time], dim=-1))

    def timesteps(self, num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, model_output: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        return x_t + (t_next - t) * model_output

    def compute_loss(self, actions: torch.Tensor, context: Context) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0])
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        return (self.denoise(x_t, t, context) - (actions - noise)) ** 2


class LinearHead(ActionHead):
    """One-shot regression head with a single projection, named like an existing policy layer."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, action_dim: int = ACTION_DIM) -> None:
        super().__init__(chunk_size, action_dim)
        self.action_out_proj = nn.Linear(CONTEXT_DIM, chunk_size * action_dim)

    def sample(self, context: Context, *, noise: torch.Tensor | None = None, num_steps: int | None = None) -> torch.Tensor:
        return self.action_out_proj(context["tokens"].mean(dim=1)).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(self, actions: torch.Tensor, context: Context) -> torch.Tensor:
        return (self.sample(context) - actions).abs()


@pytest.fixture
def head() -> ToyFlowHead:
    """Toy iterative head."""
    return ToyFlowHead()


@pytest.fixture
def context() -> Context:
    """Context with a batch of 2 and 6 tokens."""
    return {"tokens": torch.randn(2, 6, CONTEXT_DIM)}


class TestActionHead:
    """Tests for the ActionHead interface."""

    def test_forward_samples(self, context: Context) -> None:
        """Test forward samples actions and compute_loss returns a per-element loss."""
        head = LinearHead()
        actions = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(head(context), head.sample(context))
        assert head(context).shape == (2, CHUNK_SIZE, ACTION_DIM)
        assert head.compute_loss(actions, context).shape == (2, CHUNK_SIZE, ACTION_DIM)

    def test_invalid_sizes(self) -> None:
        """Test non-positive sizes are rejected."""
        with pytest.raises(ValueError, match="positive"):
            LinearHead(chunk_size=0)

    def test_base_registers_no_state(self) -> None:
        """Test inheriting from ActionHead does not change state_dict keys, so checkpoints still load."""
        plain = nn.Module()
        plain.action_out_proj = nn.Linear(CONTEXT_DIM, CHUNK_SIZE * ACTION_DIM)

        head = LinearHead()
        assert list(head.state_dict()) == list(plain.state_dict())
        head.load_state_dict(plain.state_dict())


class TestIterativeActionHead:
    """Tests for the shared sampling loop."""

    def test_integrate_matches_manual_loop(self, head: ToyFlowHead, context: Context) -> None:
        """Test integrate runs the documented denoise/step loop."""
        noise = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        timesteps = head.timesteps(3, device=noise.device, dtype=noise.dtype)

        expected = noise
        for i in range(3):
            t, t_next = timesteps[i], timesteps[i + 1]
            expected = expected + (t_next - t) * head.denoise(expected, t.expand(2), context)

        torch.testing.assert_close(head.integrate(noise, context, timesteps), expected)

    def test_sample_uses_noise_and_num_steps(self, head: ToyFlowHead, context: Context) -> None:
        """Test sample is deterministic given noise and respects num_steps."""
        noise = torch.randn(2, CHUNK_SIZE, ACTION_DIM)
        torch.testing.assert_close(head.sample(context, noise=noise), head.sample(context, noise=noise))
        assert not torch.allclose(head.sample(context, noise=noise, num_steps=1), head.sample(context, noise=noise))

    def test_prepare_context_called_once(self, head: ToyFlowHead, context: Context, mocker: MockerFixture) -> None:
        """Test prepare_context runs once per sample, not once per step."""
        spy = mocker.spy(head, "prepare_context")
        head.sample(context)
        spy.assert_called_once()
