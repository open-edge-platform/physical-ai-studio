# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Based on NVIDIA's GR00T implementation (Apache-2.0 licensed)
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""Shared neural network building blocks for policy components.

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


def _import_diffusers() -> tuple:
    """Lazy import of diffusers timestep embedding components.

    Returns:
        Tuple of (TimestepEmbedding, Timesteps) classes.

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers.models.embeddings import TimestepEmbedding, Timesteps  # noqa: PLC0415
    except ImportError as e:
        msg = "TimestepEncoder requires diffusers.\n\nInstall with:\n    pip install diffusers"
        raise ImportError(msg) from e
    else:
        return TimestepEmbedding, Timesteps


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish activation function: x * sigmoid(x).

    Args:
        x: Input tensor.

    Returns:
        Output tensor with swish activation applied.
    """
    return x * torch.sigmoid(x)


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

    def expand_action_dimension(
        self,
        old_action_dim: int,
        new_action_dim: int,
        expand_input: bool = False,  # noqa: FBT001, FBT002
        expand_output: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Expand selected weight dimensions for larger action spaces.

        Args:
            old_action_dim: Original action dimension.
            new_action_dim: New, larger action dimension.
            expand_input: Whether to expand the input dimension (dim=1).
            expand_output: Whether to expand the output dimension (dim=2).

        Raises:
            ValueError: If ``new_action_dim`` is not larger than ``old_action_dim``.
        """
        if new_action_dim <= old_action_dim:
            msg = f"New action dim {new_action_dim} must be larger than old action dim {old_action_dim}"
            raise ValueError(msg)

        if expand_input and self.W.shape[1] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_w_parts: list[torch.Tensor] = [self.W] * repeat_times
            if remainder > 0:
                new_w_parts.append(self.W[:, :remainder, :])

            self.W = nn.Parameter(torch.cat(new_w_parts, dim=1))

        if expand_output and self.W.shape[2] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_w_parts2: list[torch.Tensor] = [self.W] * repeat_times
            if remainder > 0:
                new_w_parts2.append(self.W[:, :, :remainder])

            self.W = nn.Parameter(torch.cat(new_w_parts2, dim=2))

            if self.b.shape[1] == old_action_dim:
                new_b_parts: list[torch.Tensor] = [self.b] * repeat_times
                if remainder > 0:
                    new_b_parts.append(self.b[:, :remainder])

                self.b = nn.Parameter(torch.cat(new_b_parts, dim=1))


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

    def expand_action_dimension(self, old_action_dim: int, new_action_dim: int) -> None:
        """Expand output action dimension of the category-specific MLP.

        Args:
            old_action_dim: Original action dimension.
            new_action_dim: New, larger action dimension.
        """
        # self.layer1 does not take action_dim as input, so no expansion needed
        self.layer2.expand_action_dimension(
            old_action_dim,
            new_action_dim,
            expand_input=False,
            expand_output=True,
        )


class TimestepEncoder(nn.Module):
    """Encode diffusion timesteps into embeddings via diffusers sinusoidal projection.

    Requires ``diffusers`` to be installed; the import is deferred to
    ``__init__`` so this module can be imported without diffusers.

    Args:
        embedding_dim: Output embedding dimension.
        compute_dtype: Unused; kept for API compatibility with upstream.
    """

    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32) -> None:  # noqa: ARG002
        """Initialize timestep encoder.

        Args:
            embedding_dim: Output embedding dimension.
            compute_dtype: Compute dtype (not used, for API compatibility).
        """
        super().__init__()
        timestep_embedding_cls, timesteps_cls = _import_diffusers()
        self.time_proj = timesteps_cls(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = timestep_embedding_cls(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode timesteps.

        Args:
            timesteps: Timestep indices.

        Returns:
            Timestep embeddings.
        """
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


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
            timesteps: Timesteps of shape (B,) replicated across T, or per-token
                timesteps of shape (B, T) (e.g. real-time chunking).
            cat_ids: Category/embodiment IDs of shape (B,).

        Returns:
            Encoded features of shape (B, T, hidden_size).

        Raises:
            ValueError: If timesteps shape is neither (B,) nor (B, T).
        """
        b, t, _ = actions.shape

        # Expand single scalar time across all T steps
        if timesteps.dim() == 1 and timesteps.shape[0] == b:
            timesteps = timesteps.unsqueeze(1).expand(-1, t)
        elif timesteps.dim() == 2 and timesteps.shape == (b, t):  # noqa: PLR2004
            # RLDX-1 users per-token timesteps
            pass
        else:
            msg = f"Expected `timesteps` to have shape ({b},) or ({b}, {t}); got {tuple(timesteps.shape)}."
            raise ValueError(msg)

        # Action embedding: (B, T, hidden_size)
        a_emb = self.W1(actions, cat_ids)

        # Sinusoidal encoding: (B, T, hidden_size)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)

    def expand_action_dimension(self, old_action_dim: int, new_action_dim: int) -> None:
        """Expand encoder input action dimension.

        Args:
            old_action_dim: Original action dimension.
            new_action_dim: New, larger action dimension.
        """
        # Only W1 takes action_dim as input, so only expand its input dimension
        self.W1.expand_action_dimension(
            old_action_dim,
            new_action_dim,
            expand_input=True,
            expand_output=False,
        )
