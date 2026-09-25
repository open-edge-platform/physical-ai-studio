# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Image encoder registry and implementations for Patch Policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from torch import Tensor, nn


class SupportedEncoder(StrEnum):
    """Supported image encoder backends."""

    WEBSSL = "webssl"
    DINOV2 = "dinov2"


SUPPORTED_ENCODER_NAMES = tuple(member.value for member in SupportedEncoder)

UNBATCHED_VIEW_NDIM = 4
BATCHED_VIEW_NDIM = 5


class BaseImageEncoder(nn.Module, ABC):
    """Common contract for Patch Policy image encoders.

    Subclasses implement :meth:`encode` on a flattened ``[N, C, H, W]`` batch; the shared
    ``forward`` handles shape validation and leading-dim collapse/restore.
    """

    DEFAULT_MODEL_NAME: str
    DEFAULT_OUTPUT_DIM: int
    DEFAULT_N_PATCHES: int

    def __init__(self) -> None:
        """Initialize the encoder from the subclass backbone constants."""
        super().__init__()
        self.model_name = self.DEFAULT_MODEL_NAME
        self.output_dim = self.DEFAULT_OUTPUT_DIM
        self.n_patches = self.DEFAULT_N_PATCHES

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Encode a flattened ``[N, C, H, W]`` batch into ``[N, P, E]`` tokens."""

    def forward(self, x: Tensor, goal: Tensor | None = None) -> Tensor:
        """Return patch-token embeddings.

        Accepted input shapes:
            - [B, C, H, W]
            - [B, V, C, H, W]

        Args:
            x: Image tensor.
            goal: Optional goal image tensor with the same layout as ``x``; its tokens are
                concatenated along the view dimension.

        Returns:
            Token embeddings shaped ``[B, V, P, E]``.

        Raises:
            ValueError: If the input tensor does not have 4 or 5 dimensions.
        """
        if x.ndim == UNBATCHED_VIEW_NDIM:
            x = x.unsqueeze(1)
        if x.ndim != BATCHED_VIEW_NDIM:
            msg = "Expected image tensor with shape [B, C, H, W] or [B, V, C, H, W]."
            raise ValueError(msg)

        prefix_shape = x.shape[:-3]
        tokens = self.encode(x.reshape(-1, *x.shape[-3:]))
        tokens = tokens.reshape(*prefix_shape, *tokens.shape[1:])

        if goal is not None:
            tokens = torch.cat([tokens, self.forward(goal)], dim=1)

        return tokens


class WebSSLImageEncoder(BaseImageEncoder):
    """WebSSL image encoder placeholder.

    This initial implementation makes the Patch Policy encoder contract explicit while
    keeping the real Hugging Face integration easy to swap in later.
    """

    DEFAULT_MODEL_NAME = "facebook/webssl-dino300m-full2b-224"
    DEFAULT_OUTPUT_DIM = 1024
    DEFAULT_N_PATCHES = 256

    def encode(self, x: Tensor) -> Tensor:
        """Encode a flattened ``[N, C, H, W]`` batch.

        Returns:
            Token embeddings shaped ``[N, P, E]``.
        """
        return torch.randn(x.shape[0], self.n_patches, self.output_dim, device=x.device, dtype=x.dtype)


class TimmDinoV2ImageEncoder(BaseImageEncoder):
    """Placeholder for the timm-based DINOv2 encoder backend."""

    DEFAULT_MODEL_NAME = "dinov2_vits14"
    DEFAULT_OUTPUT_DIM = 384
    DEFAULT_N_PATCHES = 256

    def encode(self, x: Tensor) -> Tensor:
        """Encode a flattened ``[N, C, H, W]`` batch.

        Returns:
            Token embeddings shaped ``[N, P, E]``.
        """
        return torch.randn(x.shape[0], self.n_patches, self.output_dim, device=x.device, dtype=x.dtype)


def resolve_image_encoder(
    encoder_name: str | SupportedEncoder | None,
) -> BaseImageEncoder:
    """Resolve a configured encoder name to a concrete encoder module.

    Args:
        encoder_name: The name of the encoder to resolve. Can be a string or a SupportedEncoder enum member.
        If None, defaults to the default encoder.

    Returns:
        An instance of the resolved encoder module.

    Raises:
        ValueError: If the encoder_name is not supported.
        NotImplementedError: If the encoder_name is supported but not yet implemented.
    """
    resolved_name = (
        (
            encoder_name.value
            if isinstance(encoder_name, SupportedEncoder)
            else (encoder_name or SupportedEncoder.WEBSSL.value)
        )
        .strip()
        .lower()
    )

    try:
        supported_encoder = SupportedEncoder(resolved_name)
    except ValueError as exc:
        supported = ", ".join(SUPPORTED_ENCODER_NAMES)
        msg = f"Unsupported encoder_name='{encoder_name}'. Supported values are: {supported}."
        raise ValueError(msg) from exc

    if supported_encoder is SupportedEncoder.WEBSSL:
        return WebSSLImageEncoder()
    if supported_encoder is SupportedEncoder.DINOV2:
        return TimmDinoV2ImageEncoder()

    msg = f"Missing implementation for encoder '{supported_encoder.value}'."
    raise NotImplementedError(msg)
