# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for image augmentation transforms."""

from __future__ import annotations

import pytest
import torch
from torchvision.transforms.v2 import ColorJitter, RandomAffine

from physicalai.transforms import DefaultImageAugmentations, RandomChoice, RandomSharpness


class TestRandomChoice:
    """Smoke tests for RandomChoice."""

    @pytest.fixture()
    def transforms(self) -> list:
        return [
            ColorJitter(brightness=(0.8, 1.2)),
            ColorJitter(contrast=(0.8, 1.2)),
            RandomAffine(degrees=5),
        ]

    def test_output_shape_preserved(self, transforms: list) -> None:
        """Output image should have the same shape as input."""
        transform = RandomChoice(transforms, n_subset=2)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_output_shape_batch(self, transforms: list) -> None:
        """Batched images should preserve shape."""
        transform = RandomChoice(transforms, n_subset=2)
        image = torch.rand(2, 3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_all_transforms_applied(self, transforms: list) -> None:
        """When n_subset=None, all transforms are applied."""
        transform = RandomChoice(transforms)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_random_order(self, transforms: list) -> None:
        """random_order=True should not change output shape."""
        transform = RandomChoice(transforms, n_subset=2, random_order=True)
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_invalid_transforms_type(self) -> None:
        """Non-sequence transforms should raise TypeError."""
        with pytest.raises(TypeError, match="sequence of callables"):
            RandomChoice(42)

    def test_invalid_n_subset(self, transforms: list) -> None:
        """n_subset out of range should raise ValueError."""
        with pytest.raises(ValueError, match="n_subset"):
            RandomChoice(transforms, n_subset=0)

    def test_mismatched_p_length(self, transforms: list) -> None:
        """p with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="Length of p"):
            RandomChoice(transforms, p=[1.0, 1.0])


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


class TestDefaultImageAugmentations:
    """Smoke tests for the ready-made augmentation pipeline."""

    def test_usable_without_arguments(self) -> None:
        """The pipeline is the one-liner alternative to spelling out the pool."""
        transform = DefaultImageAugmentations()
        assert isinstance(transform, RandomChoice)
        assert len(transform.transforms) == 6
        assert transform.n_subset == 3

    def test_output_shape_preserved(self) -> None:
        """Output image should have the same shape as input."""
        transform = DefaultImageAugmentations()
        image = torch.rand(3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_output_shape_batch(self) -> None:
        """Batched images should preserve shape."""
        transform = DefaultImageAugmentations()
        image = torch.rand(2, 3, 64, 64)
        output = transform(image)
        assert output.shape == image.shape

    def test_image_is_actually_perturbed(self) -> None:
        """The pipeline should change the image, not pass it through."""
        torch.manual_seed(0)
        transform = DefaultImageAugmentations()
        image = torch.rand(3, 64, 64)
        assert not torch.equal(transform(image), image)

    def test_n_subset_override(self) -> None:
        """n_subset controls how many transforms are applied per image."""
        transform = DefaultImageAugmentations(n_subset=1)
        assert transform.n_subset == 1
        image = torch.rand(3, 64, 64)
        assert transform(image).shape == image.shape

    def test_n_subset_out_of_range_raises(self) -> None:
        """Validation is inherited from RandomChoice."""
        with pytest.raises(ValueError, match="n_subset"):
            DefaultImageAugmentations(n_subset=7)

    def test_random_order(self) -> None:
        """random_order=True should not change output shape."""
        transform = DefaultImageAugmentations(random_order=True)
        image = torch.rand(3, 64, 64)
        assert transform(image).shape == image.shape
