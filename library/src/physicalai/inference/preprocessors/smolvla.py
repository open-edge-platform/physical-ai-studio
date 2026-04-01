# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images for SmolVLA."""

from __future__ import annotations

import cv2
import numpy as np

from physicalai.inference.constants import IMAGE_MASKS, IMAGES

from .base import Preprocessor


class ResizeSmolVLA(Preprocessor):
    """Preprocessor for resizing images for SmolVLA model using numpy operations.

    This preprocessor resizes input images to a specified resolution while maintaining
    aspect ratio through padding. It normalizes the pixel values to the range [-1, 1]
    and generates corresponding image masks.

    Attributes:
        image_resolution (tuple[int, int]): The target resolution for input images
            as (height, width). Defaults to (512, 512).
    """

    def __init__(self, image_resolution: tuple[int, int] = (512, 512)) -> None:
        """Initialize the SmolVLA numpy-based preprocessor.

        Args:
            image_resolution (tuple[int, int]): The target resolution for input images
                as (height, width). Defaults to (512, 512).
        """
        super().__init__()
        self.image_resolution = image_resolution

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and prepare images for model inference.

        Resizes images with padding, normalizes pixel values to [-1, 1] range,
        and generates corresponding attention masks.

        Args:
            inputs: Dictionary containing IMAGES key with numpy array(s) of shape
                    (height, width, channels) or list of such arrays.

        Returns:
            Dictionary with processed:
            - IMAGES: Stacked resized images of shape (batch_size, height, width, channels)
                        with pixel values normalized to [-1, 1].
            - IMAGE_MASKS: Boolean masks of shape (batch_size, height, width) indicating
                             valid image regions (all ones for padded images).
        """
        img_keys = [key for key in inputs if key.startswith(IMAGES)]
        if len(img_keys) == 1:
            images = [inputs[img_keys[0]]]
        else:
            images = [inputs[key] for key in img_keys]

        img_masks = []
        resized_images = []

        for img in images:
            resized_img = self._resize_with_pad(img, *self.image_resolution, pad_value=0)
            resized_img = resized_img * 2.0 - 1.0
            bsize = resized_img.shape[0]
            mask = np.ones(bsize, dtype=np.bool)
            resized_images.append(resized_img)
            img_masks.append(mask)

        inputs[IMAGES] = np.stack(resized_images, axis=0)
        inputs[IMAGE_MASKS] = np.stack(img_masks, axis=0)

        return inputs

    @staticmethod
    def _resize_with_pad(img: np.ndarray, width: int, height: int, pad_value: int = -1) -> np.ndarray:
        # assume no-op when width height fits already
        img_dim = 4
        if img.ndim != img_dim:
            msg = f"(b,c,h,w) expected, but {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        # Per-image cv2 bilinear resize (matches F.interpolate align_corners=False)
        batch = []
        for i in range(img.shape[0]):
            # cv2.resize expects (H, W, C) so transpose from (C, H, W)
            hwc = np.transpose(img[i], (1, 2, 0))
            resized_hwc = cv2.resize(hwc, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
            if resized_hwc.ndim == 2:
                resized_hwc = resized_hwc[:, :, np.newaxis]
            batch.append(np.transpose(resized_hwc, (2, 0, 1)))
        resized_img = np.stack(batch, axis=0)

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        if pad_height > 0 or pad_width > 0:
            padded = np.full(
                (resized_img.shape[0], resized_img.shape[1], resized_height + pad_height, resized_width + pad_width),
                fill_value=pad_value,
                dtype=resized_img.dtype,
            )
            padded[:, :, pad_height:, pad_width:] = resized_img
            return padded
        return resized_img
