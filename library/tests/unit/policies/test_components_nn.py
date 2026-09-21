# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared neural-network policy components."""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.components.nn import (
    CategorySpecificLinear,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    TimestepEncoder,
    swish,
)


class TestSharedNNPrimitives:
    """Tests for reusable components used by RLDX-1 and other policies."""

    def test_swish_activation(self) -> None:
        """Test swish computes x * sigmoid(x) and preserves gradients."""
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        out = swish(x)
        expected = x.detach() * torch.sigmoid(x.detach())
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        assert x.grad is not None

    def test_sinusoidal_encoding(self) -> None:
        """Test sinusoidal positional encoding is deterministic and timestep-sensitive."""
        encoder = SinusoidalPositionalEncoding(embedding_dim=128)
        timesteps = torch.tensor([[0.0, 100.0, 500.0]])

        out = encoder(timesteps)
        assert out.shape == (1, 3, 128)
        torch.testing.assert_close(encoder(timesteps), out)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_category_specific_linear(self) -> None:
        """Test category-specific linear layer routes by category ID."""
        layer = CategorySpecificLinear(num_categories=4, input_dim=16, hidden_dim=32)
        x = torch.randn(2, 5, 16, requires_grad=True)

        out1 = layer(x, torch.tensor([0, 0]))
        out2 = layer(x, torch.tensor([1, 1]))

        assert out1.shape == (2, 5, 32)
        assert not torch.allclose(out1, out2)
        out1.sum().backward()
        assert x.grad is not None

    def test_timestep_encoder(self) -> None:
        """Test shared timestep encoder produces distinct embeddings."""
        encoder = TimestepEncoder(embedding_dim=256)

        out1 = encoder(torch.tensor([0]))
        out2 = encoder(torch.tensor([500]))

        assert out1.shape == (1, 256)
        assert not torch.allclose(out1, out2)

    def test_multi_embodiment_action_encoder(self) -> None:
        """Test action encoder combines actions, timesteps, and embodiments."""
        encoder = MultiEmbodimentActionEncoder(action_dim=7, hidden_size=64, num_embodiments=4)
        actions = torch.randn(2, 10, 7)
        embodiment_ids = torch.tensor([0, 2])

        out1 = encoder(actions, torch.tensor([0, 0]), embodiment_ids)
        out2 = encoder(actions, torch.tensor([500, 500]), embodiment_ids)

        assert out1.shape == (2, 10, 64)
        assert not torch.allclose(out1, out2)
