# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SigLIP vision encoder for Pi0/Pi0.5 models.

This module provides the SigLIP (Sigmoid Loss for Language Image Pre-Training)
vision encoder used by PaliGemma for image understanding.

Uses HuggingFace transformers SiglipVisionModel under the hood.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SigLIPEncoder(nn.Module):
    """SigLIP vision encoder wrapper.

    Wraps the HuggingFace SiglipVisionModel for use in Pi0/Pi0.5 models.
    Outputs patch embeddings suitable for the language model.

    Args:
        model_name: HuggingFace model name for SigLIP model.
        output_hidden_states: Whether to output hidden states.

    Example:
        >>> encoder = SigLIPEncoder()
        >>> images = torch.randn(2, 3, 224, 224)
        >>> embeddings = encoder(images)
        >>> print(embeddings.shape)  # (2, 256, hidden_dim)
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        *,
        output_hidden_states: bool = False,
    ) -> None:
        """Initialize SigLIP encoder.

        Args:
            model_name: HuggingFace model name for SigLIP model.
            output_hidden_states: Whether to output hidden states.
        """
        super().__init__()
        self.model_name = model_name
        self.output_hidden_states = output_hidden_states

        # Lazy load the model
        self._vision_model: nn.Module | None = None
        self._hidden_size: int | None = None

    def _ensure_loaded(self) -> None:
        """Lazy load the vision model.

        Raises:
            ImportError: If transformers library is not installed.
        """
        if self._vision_model is not None:
            return

        try:
            from transformers import SiglipVisionModel  # noqa: PLC0415
        except ImportError as e:
            msg = "SigLIP requires transformers. Install with: uv pip install transformers"
            raise ImportError(msg) from e

        logger.info("Loading SigLIP vision model: %s", self.model_name)
        self._vision_model = SiglipVisionModel.from_pretrained(self.model_name, revision="main")
        self._hidden_size = self._vision_model.config.hidden_size

    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the vision model.

        Raises:
            RuntimeError: If vision model failed to load.
        """
        self._ensure_loaded()
        if self._hidden_size is None:
            msg = "Vision model not loaded"
            raise RuntimeError(msg)
        return self._hidden_size

    @property
    def vision_model(self) -> nn.Module:
        """Get the underlying vision model.

        Raises:
            RuntimeError: If vision model failed to load.
        """
        self._ensure_loaded()
        if self._vision_model is None:
            msg = "Vision model not loaded"
            raise RuntimeError(msg)
        return self._vision_model

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool | None = None,
    ) -> torch.Tensor:
        """Encode images to patch embeddings.

        Args:
            pixel_values: Image tensor of shape (batch, channels, height, width).
                Expected to be normalized to [-1, 1] or [0, 1] depending on model.
            output_hidden_states: Override instance setting for hidden states.

        Returns:
            Patch embeddings of shape (batch, num_patches, hidden_size).
        """
        self._ensure_loaded()

        output_hidden = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden,
            return_dict=True,
        )

        return outputs.last_hidden_state

    @staticmethod
    def get_num_patches(image_size: int = 224, patch_size: int = 14) -> int:
        """Calculate number of patches for given image size.

        Args:
            image_size: Input image size (assumes square).
            patch_size: Patch size used by the model.

        Returns:
            Number of patches (without CLS token).
        """
        return (image_size // patch_size) ** 2


def resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize images to target size with padding to preserve aspect ratio.

    This is the standard preprocessing for SigLIP/PaliGemma images.

    Args:
        images: Image tensor of shape (batch, channels, h, w) or (channels, h, w).
        height: Target height.
        width: Target width.
        mode: Interpolation mode for resizing.

    Returns:
        Resized images of shape (..., channels, height, width).
    """
    import torch.nn.functional as F  # noqa: N812, PLC0415

    # Handle single image (3D tensor: C, H, W)
    single_image_ndim = 3
    squeeze = False
    if images.ndim == single_image_ndim:
        images = images.unsqueeze(0)
        squeeze = True

    _, _, h, w = images.shape

    # Calculate scaling and new dimensions
    scale = min(height / h, width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    align_corners = None if mode == "nearest" else False

    # Resize and pad
    resized = F.interpolate(images, size=(new_h, new_w), mode=mode, align_corners=align_corners)
    pad_h, pad_w = height - new_h, width - new_w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    padded = F.pad(resized, padding, mode="constant", value=0)

    return padded.squeeze(0) if squeeze else padded


def normalize_image(
    images: torch.Tensor,
    mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    std: tuple[float, ...] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Normalize images with mean and std.

    Args:
        images: Image tensor of shape (..., channels, height, width).
            Expected to be in [0, 1] range.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Normalized images.
    """
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(-1, 1, 1)
    return (images - mean_tensor) / std_tensor
