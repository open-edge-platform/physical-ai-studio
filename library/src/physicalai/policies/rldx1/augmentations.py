# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Deterministic image geometry for the RLDX-1 policy (torchvision-only).

:class:`AspectAreaResizeAndCrop` resizes an image so its area matches a target
budget (preserving aspect ratio), then center-crops both dimensions down to a
multiple of ``m_alignment``. The same deterministic transform is used at train
and eval time; there is no train-time stochastic augmentation (crop, rotation,
color jitter).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F  # noqa: N812


class AspectAreaResizeAndCrop(nn.Module):
    """Aspect-preserving area-budget resize + ``m``-aligned center crop.

    Resizes the input so its area matches ``target_area`` (or ``min_area``
    when the input is smaller than that floor), preserving aspect ratio, then
    center-crops both dimensions down to the nearest multiple of
    ``m_alignment``. Works on PIL images or ``(..., H, W)`` tensors.
    """

    def __init__(
        self,
        target_area: float,
        m_alignment: int = 32,
        interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
        min_area: float | None = None,
    ) -> None:
        """Store the area budget, alignment multiple, interpolation and optional upscale floor."""
        super().__init__()
        self.target_area = target_area
        self.m_alignment = m_alignment
        self.interpolation = interpolation
        self.min_area = min_area

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """Resize ``inpt`` to the area budget then center-crop to an aligned size.

        Returns:
            The resized and cropped image (same type as ``inpt``).
        """
        orig_h, orig_w = F.get_size(inpt)

        # 1. Compute target scale: downscale only if over target_area, upscale only up to
        # min_area (never all the way to target_area); otherwise leave the size untouched.
        current_area = orig_h * orig_w
        min_area = self.min_area
        if min_area is not None and current_area < min_area:
            scale = math.sqrt(min_area / current_area)
        elif current_area > self.target_area:
            scale = math.sqrt(self.target_area / current_area)
        else:
            scale = 1.0

        resized_h = round(orig_h * scale)
        resized_w = round(orig_w * scale)

        # 2. Resize maintaining aspect ratio.
        inpt = F.resize(inpt, size=[resized_h, resized_w], interpolation=self.interpolation, antialias=True)

        # 3. Compute m-aligned center crop dimensions.
        crop_h = (resized_h // self.m_alignment) * self.m_alignment
        crop_w = (resized_w // self.m_alignment) * self.m_alignment

        # 4. Perform the center crop.
        return F.center_crop(inpt, output_size=[crop_h, crop_w])
