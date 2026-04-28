# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import torch

from physicalai.inference.constants import IMAGE_MASKS, IMAGES
from physicalai.inference.preprocessors import Preprocessor, ResizeSmolVLA
from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor


class TestResizeSmolVLAInit:
    def test_is_preprocessor(self) -> None:
        prep = ResizeSmolVLA()
        assert isinstance(prep, Preprocessor)

    def test_default_resolution(self) -> None:
        prep = ResizeSmolVLA()
        assert prep.image_resolution == (512, 512)

    def test_custom_resolution(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(256, 256))
        assert prep.image_resolution == (256, 256)


class TestResizeSmolVLACall:
    def test_output_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert IMAGES in result
        assert IMAGE_MASKS in result

    def test_output_shape(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 32, 32).astype(np.float32)
        result = prep({IMAGES: img})
        # 1 image → stacked with extra dim: (1, batch, channels, H, W)
        assert result[IMAGES].shape[2] == 3
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_pixel_range_normalised(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.ones((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 1.0 → 1.0 * 2 - 1 = 1.0
        np.testing.assert_allclose(result[IMAGES].max(), 1.0, atol=1e-5)

    def test_pixel_range_zeros(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.zeros((1, 3, 64, 64), dtype=np.float32)
        result = prep({IMAGES: img})
        # input 0.0 → 0.0 * 2 - 1 = -1.0
        np.testing.assert_allclose(result[IMAGES].min(), -1.0, atol=1e-5)

    def test_masks_are_boolean_ones(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(2, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGE_MASKS].dtype == np.bool_
        assert result[IMAGE_MASKS].all()

    def test_preserves_other_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = prep({IMAGES: img, "task": "pick up"})
        assert result["task"] == "pick up"

    def test_multiple_image_keys(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        inputs = {
            f"{IMAGES}.0": np.random.rand(1, 3, 64, 64).astype(np.float32),
            f"{IMAGES}.1": np.random.rand(1, 3, 64, 64).astype(np.float32),
        }
        result = prep(inputs)
        # Two images stacked
        assert result[IMAGES].shape[0] == 2

    def test_non_square_image_padded(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        # Wide image: height < width
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = prep({IMAGES: img})
        assert result[IMAGES].shape[3] == 64
        assert result[IMAGES].shape[4] == 64

    def test_dict_images_stacked(self) -> None:
        prep = ResizeSmolVLA(image_resolution=(64, 64))
        inputs = {
            IMAGES: {
                "top": np.random.rand(1, 3, 48, 48).astype(np.float32),
                "wrist": np.random.rand(1, 3, 32, 64).astype(np.float32),
            },
        }
        result = prep(inputs)
        assert result[IMAGES].shape == (2, 1, 3, 64, 64)
        assert result[IMAGE_MASKS].shape == (2, 1)
        assert result[IMAGE_MASKS].all()

    def test_consistency_vs_ref(self) -> None:
        input_batch = {
        "images": np.random.rand(1, 3, 640, 640).astype(np.float32),
        "state": np.random.rand(1, 2).astype(np.float32),
        "task": ["sample_task"]
        }

        input_batch_torch = {
            "images": torch.from_numpy(input_batch["images"]),
            "state": torch.from_numpy(input_batch["state"]),
            "task": input_batch["task"]
        }
        resolution = (512, 512)
        np_resize = ResizeSmolVLA(image_resolution=resolution)
        torch_resize = SmolVLAPreprocessor(image_resolution=resolution)

        np_result = np_resize(input_batch)
        torch_result = torch_resize(input_batch_torch)

        assert np.linalg.norm(np_result["images"] - torch_result["images"].numpy()) < 1e-1


class TestResizeSmolVLAResizeWithPad:
    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="expected"):
            ResizeSmolVLA._resize_with_pad(np.zeros((3, 64, 64)), 64, 64)

    def test_no_pad_when_matching(self) -> None:
        img = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape == (1, 3, 64, 64)

    def test_pads_to_target_size(self) -> None:
        img = np.random.rand(1, 3, 32, 64).astype(np.float32)
        result = ResizeSmolVLA._resize_with_pad(img, 64, 64)
        assert result.shape[2] == 64
        assert result.shape[3] == 64
