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
            target_dict[key] = self._resize_max(target_dict[key], *self.image_resolution)

        if not is_flat:
            batch[IMAGES] = target_dict

        return batch

    @staticmethod
    def _resize_max(img: torch.Tensor, max_width: int, max_height: int) -> torch.Tensor:
        """Resize an image to fit within the specified maximum width and height while maintaining aspect ratio.
        """
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        if cur_height <= max_height and cur_width <= max_width:
            return img

        ratio = max(cur_width / max_width, cur_height / max_height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        return F.interpolate(
            img,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )


