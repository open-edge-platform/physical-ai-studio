# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Based on NVIDIA's GR00T implementation (Apache-2.0 licensed)
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""Neural network building blocks for Groot policy.

This module contains reusable primitive components:
- Activations: swish
- Encodings: SinusoidalPositionalEncoding
- Layers: CategorySpecificLinear, CategorySpecificMLP
- Encoders: MultiEmbodimentActionEncoder
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# =============================================================================
# Activations
# =============================================================================


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function: x * sigmoid(x).

    Args:
        x: Input tensor.

    Returns:
        Output tensor with swish activation applied.
    """
    return x * torch.sigmoid(x)


# =============================================================================
# Positional Encodings
# =============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timesteps.

    Produces a sinusoidal encoding of shape (B, T, embedding_dim)
    given timesteps of shape (B, T).

    Args:
        embedding_dim: Dimension of the embedding output.

    Examples:
        >>> encoder = SinusoidalPositionalEncoding(embedding_dim=256)
        >>> timesteps = torch.tensor([[0, 1, 2], [3, 4, 5]])  # (B=2, T=3)
        >>> encoding = encoder(timesteps)  # (2, 3, 256)
    """

    def __init__(self, embedding_dim: int) -> None:
        """Initialize sinusoidal positional encoding.

        Args:
            embedding_dim: Output embedding dimension.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal encoding.

        Args:
            timesteps: Timestep indices of shape (B, T).

        Returns:
            Sinusoidal encoding of shape (B, T, embedding_dim).
        """
        timesteps = timesteps.float()

        _b, _t = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # Log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        return torch.cat([sin, cos], dim=-1)  # (B, T, embedding_dim)


# =============================================================================
# Category-Specific Layers (Multi-Embodiment Support)
# =============================================================================


class CategorySpecificLinear(nn.Module):
    """Linear layer with per-category (embodiment) weights.

    Enables multi-embodiment support by having separate weight matrices
    for each category/embodiment type.

    Args:
        num_categories: Number of distinct embodiment categories.
        input_dim: Input feature dimension.
        hidden_dim: Output feature dimension.
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        """Initialize category-specific linear layer.

        Args:
            num_categories: Number of distinct categories.
            input_dim: Input feature dimension.
            hidden_dim: Output feature dimension.
        """
        super().__init__()
        self.num_categories = num_categories
        # For each category, separate weights and biases
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with category-specific weights.

        Args:
            x: Input tensor of shape (B, T, input_dim).
            cat_ids: Category/embodiment IDs of shape (B,).

        Returns:
            Output tensor of shape (B, T, hidden_dim).
        """
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with per-category weights.

    Args:
        num_categories: Number of embodiment categories.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize category-specific 2-layer MLP.

        Args:
            num_categories: Number of distinct categories.
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output feature dimension.
        """
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2-layer category-specific MLP.

        Args:
            x: Input tensor of shape (B, T, input_dim).
            cat_ids: Category/embodiment IDs of shape (B,).

        Returns:
            Output tensor of shape (B, T, output_dim).
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


# =============================================================================
# Action Encoders
# =============================================================================


class MultiEmbodimentActionEncoder(nn.Module):
    """Encodes actions and timesteps for multi-embodiment settings.

    Combines action features with sinusoidal timestep encoding,
    using category-specific linear layers.

    Args:
        action_dim: Dimension of action vectors.
        hidden_size: Hidden dimension for processing.
        num_embodiments: Number of embodiment categories.
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        """Initialize multi-embodiment action encoder.

        Args:
            action_dim: Dimension of action vectors.
            hidden_size: Hidden dimension for processing.
            num_embodiments: Number of embodiment categories.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{d -> w}, W2: R^{2w -> w}, W3: R^{w -> w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode actions with timestep information.

        Args:
            actions: Action tensor of shape (B, T, action_dim).
            timesteps: Timesteps of shape (B,) - single scalar per batch.
            cat_ids: Category/embodiment IDs of shape (B,).

        Returns:
            Encoded features of shape (B, T, hidden_size).

        Raises:
            ValueError: If timesteps shape is not (B,).
        """
        b, t, _ = actions.shape

        # Expand single scalar time across all T steps
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            timesteps = timesteps.unsqueeze(1).expand(-1, t)  # (B,) -> (B, T)
        else:
            msg = "Expected `timesteps` to have shape (B,) to replicate across T."
            raise ValueError(msg)

        # Action embedding: (B, T, hidden_size)
        a_emb = self.W1(actions, cat_ids)

        # Sinusoidal encoding: (B, T, hidden_size)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # Concat and process: (B, T, 2*hidden_size) -> (B, T, hidden_size)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        return self.W3(x, cat_ids)
