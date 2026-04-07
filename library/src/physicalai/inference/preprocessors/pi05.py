# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessor that resizes images and builds tokenized prompts for Pi0.5."""

from __future__ import annotations

import cv2
import numpy as np

from physicalai.inference.constants import IMAGE_MASKS, IMAGES, STATE, TASK

from .base import Preprocessor

# Number of discretization bins for state values.
_NUM_BINS = 256


class Pi05Preprocessor(Preprocessor):
    r"""Preprocessor for Pi0.5 model: image resize + text prompt construction.

    Resizes input images to a target resolution with aspect-ratio-preserving padding,
    normalizes pixel values to [-1, 1], discretizes the state vector into 256 bins,
    and builds a ``"Task: …, State: …;\\nAction: "`` prompt.

    Tokenization is handled separately (e.g. by ``HFTokenizer``).

    Args:
        image_resolution: Target (height, width) for images.
        empty_cameras: Number of extra -1-filled camera slots to append.
    """

    def __init__(
        self,
        image_resolution: tuple[int, int] = (224, 224),
        empty_cameras: int = 0,
    ) -> None:
        """Initialize the Pi0.5 preprocessor."""
        super().__init__()
        self._image_resolution = image_resolution
        self._empty_cameras = empty_cameras

    def __call__(self, inputs: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray | list[str]]:
        """Preprocess images and text for Pi0.5 inference.

        Args:
            inputs: Dict with image keys (``images`` or ``images.*``),
                a ``task`` key (list of strings), and a state key
                (numpy array of shape ``(batch, state_dim)``).

        Returns:
            Updated dict with:
            - ``images``: stacked ``(n_cameras, batch, C, H, W)`` float32 array
            - ``image_masks``: stacked ``(n_cameras, batch)`` bool array
            - ``task``: list of formatted prompt strings
        """
        # --- images ---
        images, img_masks = self._preprocess_images(inputs)

        if self._empty_cameras > 0 and images:
            for _ in range(self._empty_cameras):
                images.append(np.full_like(images[-1], -1.0))
                img_masks.append(np.zeros_like(img_masks[-1]))

        if images:
            inputs[IMAGES] = np.stack(images, axis=0)
            inputs[IMAGE_MASKS] = np.stack(img_masks, axis=0)

        # --- text prompt ---
        state: np.ndarray = inputs[STATE]
        if state.ndim > 2:  # noqa: PLR2004
            state = state[:, -1, :]

        bins = np.linspace(-1, 1, _NUM_BINS + 1)[:-1]
        discretized = np.digitize(state, bins) - 1

        task = inputs.get(TASK)
        if task is None:
            task = [""] * state.shape[0]
        elif isinstance(task, str):
            task = [task]

        prompts: list[str] = []
        for i, t in enumerate(task):
            cleaned = t.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[i]))
            prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")

        inputs[TASK] = prompts

        return inputs

    # ------------------------------------------------------------------
    # image helpers
    # ------------------------------------------------------------------

    def _preprocess_images(
        self,
        inputs: dict[str, np.ndarray | list[str]],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Resize, pad, and normalize all camera images found in *inputs*.

        Iterates over all keys that start with ``images`` (excluding ``is_pad``
        variants), extracts the latest timestep when a temporal dimension is
        present, converts to float32, resizes to ``self._image_resolution`` with
        centre-padding, and normalises from ``[0, 1]`` to ``[-1, 1]``.

        Args:
            inputs: Preprocessor input dict. Image arrays are expected to have
                shape ``(B, C, H, W)`` or ``(B, T, C, H, W)``.

        Returns:
            Tuple of:
            - ``images``: list of ``(B, C, H, W)`` float32 arrays, one per camera.
            - ``masks``: list of ``(B,)`` bool arrays (all ``True`` for real cameras).
        """
        img_keys = sorted(k for k in inputs if k.startswith(IMAGES) and "is_pad" not in k)
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        max_image_dim = 5
        for key in img_keys:
            img: np.ndarray = inputs[key]
            if img.ndim == max_image_dim:
                img = img[:, -1, :, :, :]

            if img.dtype != np.float32:
                img = img.astype(np.float32)

            # Detect layout: assume channels-first when dim-1 == 3
            channels_first = img.shape[1] == 3  # noqa: PLR2004

            if channels_first:
                img = np.transpose(img, (0, 2, 3, 1))  # -> (B, H, W, C)

            h, w = img.shape[1], img.shape[2]
            target_h, target_w = self._image_resolution
            if (h, w) != (target_h, target_w):
                img = self._resize_with_pad(img, target_w, target_h, pad_value=0)

            # [0, 1] -> [-1, 1]
            img = img * 2.0 - 1.0

            if channels_first:
                img = np.transpose(img, (0, 3, 1, 2))  # -> (B, C, H, W)

            bsize = img.shape[0]
            mask = np.ones(bsize, dtype=np.bool_)
            images.append(img)
            masks.append(mask)

        return images, masks

    @staticmethod
    def _resize_with_pad(  # noqa: PLR0914
        img: np.ndarray,
        width: int,
        height: int,
        pad_value: int = 0,
    ) -> np.ndarray:
        """Resize a ``(B, H, W, C)`` array with centre-padding.

        Args:
            img: batch of images in HWC layout, shape ``(B, H, W, C)``.
            width: target width.
            height: target height.
            pad_value: fill value for padded pixels.

        Returns:
            Resized and padded array of shape ``(B, height, width, C)``.

        Raises:
            ValueError: If *img* does not have 4 dimensions.
        """
        expected_ndim = 4
        if img.ndim != expected_ndim:
            msg = f"(B, H, W, C) expected, but got shape {img.shape}"
            raise ValueError(msg)

        cur_height, cur_width = img.shape[1], img.shape[2]

        ratio = max(cur_width / width, cur_height / height)
        resized_h = int(cur_height / ratio)
        resized_w = int(cur_width / ratio)

        batch = []
        for i in range(img.shape[0]):
            resized = cv2.resize(
                img[i],
                (resized_w, resized_h),
                interpolation=cv2.INTER_LINEAR,
            )
            if resized.ndim == 2:  # noqa: PLR2004
                resized = resized[:, :, np.newaxis]
            batch.append(resized)
        resized_img = np.stack(batch, axis=0)

        pad_h0, remainder_h = divmod(height - resized_h, 2)
        pad_h1 = pad_h0 + remainder_h
        pad_w0, remainder_w = divmod(width - resized_w, 2)
        pad_w1 = pad_w0 + remainder_w

        if pad_h0 + pad_h1 > 0 or pad_w0 + pad_w1 > 0:
            padded = np.full(
                (resized_img.shape[0], height, width, resized_img.shape[3]),
                fill_value=pad_value,
                dtype=resized_img.dtype,
            )
            padded[:, pad_h0 : pad_h0 + resized_h, pad_w0 : pad_w0 + resized_w, :] = resized_img
            return padded
        return resized_img
