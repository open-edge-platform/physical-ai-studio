# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for RLDX-1 export image-resolution derivation."""

from __future__ import annotations

import pytest

from physicalai.data import FeatureType
from physicalai.policies.rldx1.utils.export import export_image_resolution_after_preprocessor_from_stats


def _stats_with_visual_shape(height: int, width: int) -> dict[str, dict[str, object]]:
    return {
        "observation.images.front": {
            "type": FeatureType.VISUAL,
            "shape": [3, height, width],
        },
    }


def test_export_resolution_matches_no_resize_path() -> None:
    """When area already matches budget, aligned size is unchanged."""
    stats = _stats_with_visual_shape(256, 256)

    image_resolution = export_image_resolution_after_preprocessor_from_stats(
        stats,
        image_max_area=65536,
        image_resize_m=32,
        image_min_area=None,
    )

    assert image_resolution == (256, 256)


def test_export_resolution_matches_downscale_and_align() -> None:
    """Downscale-by-area then m-align center-crop mirrors preprocessor geometry."""
    stats = _stats_with_visual_shape(480, 640)

    image_resolution = export_image_resolution_after_preprocessor_from_stats(
        stats,
        image_max_area=65536,
        image_resize_m=32,
        image_min_area=None,
    )

    # scale = sqrt(65536 / (480*640)) => resized ~= (222, 296) => aligned (192, 288)
    assert image_resolution == (192, 288)


def test_export_resolution_matches_min_area_upscale() -> None:
    """Upscale to min_area then align for tiny images."""
    stats = _stats_with_visual_shape(96, 96)

    image_resolution = export_image_resolution_after_preprocessor_from_stats(
        stats,
        image_max_area=65536,
        image_resize_m=32,
        image_min_area=65536,
    )

    assert image_resolution == (256, 256)


def test_export_resolution_raises_without_visual_features() -> None:
    """Missing visual stats entries should fail loudly."""
    stats = {
        "observation.state": {
            "type": FeatureType.STATE,
            "shape": [7],
        },
    }

    with pytest.raises(RuntimeError, match="Failed to determine image resolution"):
        export_image_resolution_after_preprocessor_from_stats(
            stats,
            image_max_area=65536,
            image_resize_m=32,
            image_min_area=None,
        )
