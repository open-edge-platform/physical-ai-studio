# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 VAE helpers for LingBot-VA.

The VAE is the stock diffusers ``AutoencoderKLWan`` (``z_dim=48``, temporal downsample 4).
It is frozen and *not* stored in the policy checkpoint; it is pulled from
``config.wan_pretrained_path`` on first use.

:class:`WanVAEStreamingWrapper` drives the encoder in causal streaming mode so successive
observation chunks share the encoder's temporal ``feat_cache`` — that is what makes the
per-chunk latent frame counts line up during an autoregressive rollout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


def _lazy_import_autoencoder() -> Any:  # noqa: ANN401
    """Import ``AutoencoderKLWan`` from diffusers.

    Returns:
        The ``AutoencoderKLWan`` class.

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers import AutoencoderKLWan  # noqa: PLC0415
    except ImportError as e:
        msg = "LingBot-VA requires diffusers.\n\nInstall with:\n    uv pip install 'physicalai-train[lingbot_va]'"
        raise ImportError(msg) from e
    return AutoencoderKLWan


def load_vae(
    vae_path: str,
    torch_dtype: torch.dtype,
    torch_device: str | torch.device,
    subfolder: str | None = None,
) -> Any:  # noqa: ANN401
    """Load the frozen Wan2.2 VAE.

    Args:
        vae_path: HuggingFace repo id or local directory.
        torch_dtype: Dtype to load the weights in.
        torch_device: Device to place the VAE on.
        subfolder: Sub-folder inside ``vae_path`` holding the VAE.

    Returns:
        The loaded ``AutoencoderKLWan``.
    """
    autoencoder = _lazy_import_autoencoder()
    vae = autoencoder.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch_dtype)  # nosec B615
    return vae.to(torch_device)


def patchify(x: torch.Tensor, patch_size: int | None) -> torch.Tensor:
    """Space-to-depth the VAE input when the checkpoint uses a patchified encoder.

    Args:
        x: Video tensor of shape ``[B, C, F, H, W]``.
        patch_size: Spatial patch size, or ``None``/``1`` for no patchification.

    Returns:
        The (possibly) patchified tensor.
    """
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)


def normalize_latents(
    mu: torch.Tensor,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
) -> torch.Tensor:
    """Channel-normalize raw VAE latents to the model's latent space.

    Args:
        mu: Raw latent tensor of shape ``[B, C, F, H, W]``.
        latents_mean: Per-channel latent means from the VAE config.
        latents_std: Per-channel latent standard deviations from the VAE config.

    Returns:
        Normalized latents, cast back to ``mu``'s dtype.
    """
    mean = torch.tensor(latents_mean).view(1, -1, 1, 1, 1).to(mu.device)
    inv_std = (1.0 / torch.tensor(latents_std)).view(1, -1, 1, 1, 1).to(mu.device)
    return ((mu.float() - mean) * inv_std).to(mu)


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    z_dim: int,
) -> torch.Tensor:
    """Invert :func:`normalize_latents` so predicted latents can be VAE-decoded.

    Args:
        latents: Normalized latents of shape ``[B, z_dim, F, H, W]``.
        latents_mean: Per-channel latent means from the VAE config.
        latents_std: Per-channel latent standard deviations from the VAE config.
        z_dim: Number of latent channels.

    Returns:
        Latents in the VAE's own scale.
    """
    mean = torch.tensor(latents_mean).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    return latents / inv_std + mean


class WanVAEStreamingWrapper:
    """Drive an ``AutoencoderKLWan`` encoder in causal streaming mode.

    Successive calls to :meth:`encode_chunk` share the encoder's causal ``feat_cache``, so
    a chunk of ``F`` observed frames collapses to ``F / 4`` latent frames that continue the
    previous chunk's temporal context. :meth:`clear_cache` must be called between episodes.

    Args:
        vae_model: The frozen ``AutoencoderKLWan`` to wrap.
    """

    def __init__(self, vae_model: Any) -> None:  # noqa: ANN401
        """Wrap the VAE and size the causal feature cache from its encoder."""
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        cached_counts = getattr(self.vae, "_cached_conv_counts", None)
        if cached_counts is not None:
            self.enc_conv_num = cached_counts["encoder"]
        else:
            self.enc_conv_num = sum(1 for m in self.encoder.modules() if m.__class__.__name__ == "WanCausalConv3d")

        self.feat_cache: list[torch.Tensor | None] = []
        self.clear_cache()

    def clear_cache(self) -> None:
        """Reset the causal feature cache (call between episodes)."""
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode one temporal chunk, carrying the causal cache across calls.

        Args:
            x_chunk: Video chunk of shape ``[B, C, F, H, W]`` scaled to ``[-1, 1]``.

        Returns:
            The pre-sampling encoder output ``[B, 2 * z_dim, F // 4, H // 8, W // 8]``.
        """
        patch_size = getattr(self.vae.config, "patch_size", None)
        x_chunk = patchify(x_chunk, patch_size)
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=[0])
        return self.quant_conv(out)
