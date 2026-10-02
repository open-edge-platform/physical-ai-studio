# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action head registry and contract for Patch Policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from .config import PatchPolicyConfig

TOKEN_NDIM = 4


class SupportedActionHead(StrEnum):
    """Supported action head backends."""

    VQBET = "vqbet"
    DIFFUSION = "diffusion"


SUPPORTED_ACTION_HEAD_NAMES = tuple(member.value for member in SupportedActionHead)


class BaseActionHead(nn.Module, ABC):
    """Common contract for Patch Policy action heads.

    Heads consume ``[B, T, P, E]`` patch tokens (``T`` observation steps, ``P`` tokens per
    step, ``E`` token dim) and produce ``[B, T, W, A]`` action chunks.
    """

    def __init__(self, token_dim: int, act_dim: int, chunk_size: int) -> None:
        """Initialize the shared head state."""
        super().__init__()
        self.token_dim = token_dim
        self.act_dim = act_dim
        self.chunk_size = chunk_size

    @abstractmethod
    def compute_loss(self, tokens: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the training loss from ``[B, T, P, E]`` tokens and ``[B, T, W, A]`` actions."""

    @abstractmethod
    def predict(self, tokens: Tensor) -> Tensor:
        """Predict a ``[B, T, W, A]`` action chunk from ``[B, T, P, E]`` tokens."""

    def forward(
        self,
        tokens: Tensor,
        actions: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
        """Predict actions, or compute the loss when ``actions`` are supplied.

        Args:
            tokens: ``[B, T, P, E]`` patch tokens.
            actions: Optional ``[B, T, W, A]`` target action chunks.

        Returns:
            Predicted actions, or ``(loss, loss_dict)`` when ``actions`` is given.

        Raises:
            ValueError: If ``tokens`` does not have shape ``[B, T, P, E]``.
        """
        if tokens.ndim != TOKEN_NDIM:
            msg = f"Expected tokens with shape [B, T, P, E], got {tuple(tokens.shape)}."
            raise ValueError(msg)

        if actions is None:
            return self.predict(tokens)
        return self.compute_loss(tokens, actions)


class DiffusionActionHead(BaseActionHead):
    """Placeholder for the diffusion action head.

    To be ported from ``TransformerForDiffusion``: a transformer encoder over the patch
    tokens feeding a decoder over the noisy action sequence, under a patch-aware memory mask.
    """

    def compute_loss(self, tokens: Tensor, actions: Tensor) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Not implemented yet.

        Raises:
            NotImplementedError: Always.
        """
        msg = "DiffusionActionHead is not implemented yet."
        raise NotImplementedError(msg)

    def predict(self, tokens: Tensor) -> Tensor:
        """Not implemented yet.

        Raises:
            NotImplementedError: Always.
        """
        msg = "DiffusionActionHead is not implemented yet."
        raise NotImplementedError(msg)


def resolve_action_head(
    action_head_name: str | SupportedActionHead | None,
    config: PatchPolicyConfig,
    token_dim: int,
    act_dim: int,
    n_patches: int,
) -> BaseActionHead:
    """Resolve a configured action head name to a concrete head module.

    Args:
        action_head_name: Name of the head to resolve. ``None`` selects the default head.
        config: Patch Policy config supplying the head hyperparameters.
        token_dim: Dimension of a single patch token.
        act_dim: Dimension of a single action.
        n_patches: Number of patch tokens per observation step.

    Returns:
        An instance of the resolved head module.

    Raises:
        ValueError: If ``action_head_name`` is not supported.
        NotImplementedError: If the head is supported but not yet implemented.
    """
    resolved_name = (
        (
            action_head_name.value
            if isinstance(action_head_name, SupportedActionHead)
            else (action_head_name or SupportedActionHead.VQBET.value)
        )
        .strip()
        .lower()
    )

    try:
        supported_head = SupportedActionHead(resolved_name)
    except ValueError as exc:
        supported = ", ".join(SUPPORTED_ACTION_HEAD_NAMES)
        msg = f"Unsupported action_head_name='{action_head_name}'. Supported values are: {supported}."
        raise ValueError(msg) from exc

    if supported_head is SupportedActionHead.VQBET:
        from .vqbet import VQBeTActionHead

        return VQBeTActionHead(config=config, token_dim=token_dim, act_dim=act_dim, n_patches=n_patches)
    if supported_head is SupportedActionHead.DIFFUSION:
        return DiffusionActionHead(
            token_dim=token_dim,
            act_dim=act_dim,
            chunk_size=config.chunk_size,
        )

    msg = f"Missing implementation for action head '{supported_head.value}'."
    raise NotImplementedError(msg)
