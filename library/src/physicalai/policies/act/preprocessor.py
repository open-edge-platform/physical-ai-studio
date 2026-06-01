# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor for ACT model.

This module provides preprocessing functionality for transforming observations
and actions into the format expected by ACT model.

Handles:
- Image resizing
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from physicalai.data.observation import IMAGES, Observation


class ACTPreprocessor(torch.nn.Module):
    """Preprocessor for ACT model inputs.

    - Resizes images to target resolution with keeping proportions

    Args:
        image_resolution: Target image resolution (height, width).

    Example:
        >>> preprocessor = ACTPreprocessor(
        ...     image_resolution=(512, 512),
        ... )
        >>> batch = preprocessor(raw_batch)
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (512, 512),
    ) -> None:
        """Initialize the ACT preprocessor.

        Args:
            image_resolution: Target resolution for input images as (height, width).
                Defaults to (512, 512).
        """
        super().__init__()

        self.image_resolution = image_resolution

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a batch by applying newline processing, tokenization, and normalization.

        Args:
            batch: A dictionary containing input data with keys including IMAGES.

        Returns:
            A dictionary containing the processed batch with resized images.
        """
        batch = copy(batch)
        target_keys = Observation.get_flattened_keys(batch, IMAGES)
        target_keys = [key for key in target_keys if "is_pad" not in key]
        target_dict = batch
        is_flat = True

        if IMAGES in batch and isinstance(batch[IMAGES], dict):
            target_keys = list(batch[IMAGES].keys())
            target_dict = copy(batch[IMAGES])
            is_flat = False

        for key in target_keys:
            target_dict[key] = self._resize_with_ar_pad(target_dict[key], *self.image_resolution)

        if not is_flat:
            batch[IMAGES] = target_dict

        return batch

    @staticmethod
    def _resize_with_ar_pad(img: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
        """Resize an image tensor to the target resolution while maintaining aspect ratio and padding the remainder.

        The image is scaled so it fits within the target resolution, then zero-padded
        symmetrically to exactly match (target_height, target_width).

        Args:
            img (torch.Tensor): Input image tensor with shape (batch, channels, height, width).
            target_width (int): Target width for the resized image.
            target_height (int): Target height for the resized image.

        Returns:
            torch.Tensor: Image tensor of shape (batch, channels, target_height, target_width),
                preserving the original aspect ratio with zero padding.

        Raises:
            ValueError: If the input tensor does not have 4 dimensions (batch, channels, height, width).
        """
        return img

        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / target_width, cur_height / target_height)
        resized_height = min(int(cur_height / ratio), target_height)
        resized_width = min(int(cur_width / ratio), target_width)

        if (resized_height, resized_width) != (cur_height, cur_width):
            img = F.interpolate(
                img,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )

        pad_height = target_height - resized_height
        pad_width = target_width - resized_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))
