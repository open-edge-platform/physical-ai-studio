# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch image patchification for MolmoAct2."""

from __future__ import annotations

import torch

_IMAGE_NDIM = 4
_NUM_CHANNELS = 3


class MolmoAct2ImageProcessor:
    """Normalize and patchify pre-resized images into model-ready crops.

    Steps:
        1. Validate BCHW image shape, dtype, and crop mode.
        2. Apply channel mean and standard-deviation normalization.
        3. Split each image into flattened vision patches.
        4. Build pooled patch indices and image-grid metadata.
    """

    def __init__(
        self,
        *,
        crop_mode: str,
        size: dict[str, int],
        patch_size: int,
        pooling_size: list[int],
        image_mean: list[float],
        image_std: list[float],
    ) -> None:
        """Read declared image settings and precompute pooling."""
        self.crop_mode = str(crop_mode)
        self.height = int(size["height"])
        self.width = int(size["width"])
        self.patch_size = int(patch_size)
        self.pool_h, self.pool_w = (int(pooling_size[0]), int(pooling_size[1]))
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        patch_h = self.height // self.patch_size
        patch_w = self.width // self.patch_size
        self._pooling, self.pooled_h, self.pooled_w = self._pooling_indices(patch_h, patch_w)

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process a batch of BCHW images into model-ready tensors.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing processed image tensors.

        Raises:
            ValueError: If images are not 3-channel BCHW tensors.
            NotImplementedError: If crop mode is not resize.
        """
        if images.ndim != _IMAGE_NDIM or images.shape[1] != _NUM_CHANNELS:
            msg = f"Expected images of shape (M, 3, H, W), got {tuple(images.shape)}."
            raise ValueError(msg)
        if self.crop_mode != "resize":
            msg = f"MolmoAct2ImageProcessor only supports crop_mode='resize', got {self.crop_mode!r}."
            raise NotImplementedError(msg)

        num_images = images.shape[0]
        pixel_values = self._patchify(self._normalize(images))
        pooling = self._pooling.to(images.device)
        image_token_pooling = pooling.unsqueeze(0).expand(num_images, -1, -1).reshape(-1, pooling.shape[-1])
        grid_row = torch.tensor([self.pooled_h, self.pooled_w, 0, 0], dtype=torch.int64, device=images.device)
        return {
            "pixel_values": pixel_values,
            "image_token_pooling": image_token_pooling,
            "image_grids": grid_row.unsqueeze(0).expand(num_images, -1).contiguous(),
            "image_num_crops": torch.ones(num_images, dtype=torch.int64, device=images.device),
        }

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply channel mean/std normalization to the input images.

        Args:
            images (torch.Tensor): A BCHW tensor of images to normalize.

        Returns:
            torch.Tensor: The normalized images.

        Raises:
            ValueError: If image dtype is unsupported.
        """
        if images.dtype not in {torch.float16, torch.float32}:
            msg = f"Expected images of dtype float16 or float32, got {images.dtype}."
            raise ValueError(msg)
        mean = torch.tensor(self.image_mean, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
        return (images - mean) / std

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        """Split BCHW pixels into flattened patches.

        Returns:
            torch.Tensor: The flattened patches of the input images.
        """
        num_images, channels, height, width = pixels.shape
        patch = self.patch_size
        pixels = pixels.permute(0, 2, 3, 1)
        pixels = pixels.reshape(num_images, height // patch, patch, width // patch, patch, channels)
        pixels = pixels.permute(0, 1, 3, 2, 4, 5)
        return pixels.reshape(num_images, (height // patch) * (width // patch), patch * patch * channels)

    def _pooling_indices(self, patch_h: int, patch_w: int) -> tuple[torch.Tensor, int, int]:
        """Build patch indices grouped per pooled token.

        Returns:
            tuple[torch.Tensor, int, int]: A tuple containing the patch indices, pooled height, and pooled width.
        """
        indices = torch.arange(patch_h * patch_w, dtype=torch.int64).reshape(patch_h, patch_w)
        pooled_h = (patch_h + self.pool_h - 1) // self.pool_h
        pooled_w = (patch_w + self.pool_w - 1) // self.pool_w
        pad_h = pooled_h * self.pool_h - patch_h
        pad_w = pooled_w * self.pool_w - patch_w
        indices = torch.nn.functional.pad(
            indices,
            (pad_w // 2, (pad_w + 1) // 2, pad_h // 2, (pad_h + 1) // 2),
            value=-1,
        )
        indices = indices.reshape(pooled_h, self.pool_h, pooled_w, self.pool_w)
        indices = indices.permute(0, 2, 1, 3).reshape(pooled_h * pooled_w, self.pool_h * self.pool_w)
        return indices, pooled_h, pooled_w
