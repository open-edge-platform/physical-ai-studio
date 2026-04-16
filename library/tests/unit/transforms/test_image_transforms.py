# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for image augmentation transforms."""

from __future__ import annotations

import pytest
import torch
from torchvision.transforms.v2 import ColorJitter, RandomAffine

from physicalai.transforms import RandomChoiceApply, RandomSharpness


class TestRandomChoiceApply:
    """Smoke tests for RandomChoiceApply."""

    @pytest.fixture()
    def transforms(self) -> list:
        return [
            ColorJitter(brightness=(0.8, 1.2)),
            ColorJitter(contrast=(0.8, 1.2)),
            RandomAffine(degrees=5),
        ]

    def test_output_shape_preserved(self, transforms: list) -> None:
        """Output image should have the same shape as input."""
        transform = RandomChoiceApply(transforms, n_subset=2)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_output_shape_batch(self, transforms: list) -> None:
        """Batched images should preserve shape."""
        transform = RandomChoiceApply(transforms, n_subset=2)
        image = torch.rand(2, 3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_all_transforms_applied(self, transforms: list) -> None:
        """When n_subset=None, all transforms are applied."""
        transform = RandomChoiceApply(transforms)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_random_order(self, transforms: list) -> None:
        """random_order=True should not change output shape."""
        transform = RandomChoiceApply(transforms, n_subset=2, random_order=True)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_invalid_transforms_type(self) -> None:
        """Non-sequence transforms should raise TypeError."""
        with pytest.raises(TypeError, match="sequence of callables"):
            RandomChoiceApply(42)

    def test_invalid_n_subset(self, transforms: list) -> None:
        """n_subset out of range should raise ValueError."""
        with pytest.raises(ValueError, match="n_subset"):
            RandomChoiceApply(transforms, n_subset=0)

    def test_mismatched_p_length(self, transforms: list) -> None:
        """p with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="Length of p"):
            RandomChoiceApply(transforms, p=[1.0, 1.0])


class TestRandomSharpness:
    """Smoke tests for RandomSharpness."""

    def test_output_shape_preserved(self) -> None:
        """Output image should have the same shape as input."""
        transform = RandomSharpness(sharpness=[0.5, 1.5])
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_output_shape_batch(self) -> None:
        """Batched images should preserve shape."""
        transform = RandomSharpness(sharpness=[0.5, 1.5])
        image = torch.rand(2, 3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_single_channel(self) -> None:
        """Grayscale images should be supported."""
        transform = RandomSharpness(sharpness=[0.5, 1.5])
        image = torch.rand(1, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_scalar_sharpness(self) -> None:
        """Single float sharpness should work."""
        transform = RandomSharpness(sharpness=0.5)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_negative_sharpness_raises(self) -> None:
        """Negative scalar sharpness should raise ValueError."""
        with pytest.raises(ValueError, match="non negative"):
            RandomSharpness(sharpness=-1.0)

    def test_invalid_range_raises(self) -> None:
        """Inverted range should raise ValueError."""
        with pytest.raises(ValueError, match="sharpness values"):
            RandomSharpness(sharpness=[1.5, 0.5])
