
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images for SmolVLA."""


from __future__ import annotations

from .base import Preprocessor
import numpy as np
from PIL import Image


class ResizeSmolVLA(Preprocessor):
    def __init__(self, image_resolution: tuple[int, int] = (512, 512)) -> None:
        super().__init__()
        self.image_resolution = image_resolution

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = inputs["images"]

        if isinstance(images, np.ndarray):
            images = [images]

        img_masks = []
        resized_images = []

        for img in images:
            img = self._resize_with_pad(img, *self.image_resolution, pad_value=0)
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            mask = np.ones(bsize, dtype=np.bool_)
            resized_images.append(img)
            img_masks.append(mask)

        inputs["images"] = np.stack(resized_images, axis=0)
        inputs["image_masks"] = np.stack(img_masks, axis=0)

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

        # Per-image PIL bilinear resize (matches F.interpolate align_corners=False)
        batch = []
        for i in range(img.shape[0]):
            channels = []
            for c in range(img.shape[1]):
                pil_img = Image.fromarray(img[i, c].astype(np.float32), mode="F")
                pil_img = pil_img.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
                channels.append(np.asarray(pil_img, dtype=img.dtype))
            batch.append(np.stack(channels, axis=0))
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
