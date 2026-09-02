# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers shared across gym wrappers."""

from collections.abc import Mapping
from typing import Any


def validate_camera_name_mapping(
    mapping: Mapping[str, str] | None,
    camera_keys: list[str],
) -> dict[str, str]:
    """Validate a mapping from existing image keys to output image keys.

    Returns:
        A copy of the validated mapping, or an empty mapping when no mapping is provided.

    Raises:
        ValueError: If source keys are unknown, output keys are empty or non-string, or
            the mapping creates duplicate output keys.
    """
    if mapping is None:
        return {}

    unknown_keys = sorted(set(mapping) - set(camera_keys))
    if unknown_keys:
        msg = (
            "camera_name_mapping keys must map from existing output keys. "
            f"Unknown keys: {unknown_keys}. Existing keys: {sorted(set(camera_keys))}"
        )
        raise ValueError(msg)

    if any(not isinstance(value, str) or not value for value in mapping.values()):
        msg = "camera_name_mapping values must be non-empty strings."
        raise ValueError(msg)

    mapped_keys = [mapping.get(key, key) for key in camera_keys]
    if len(mapped_keys) != len(set(mapped_keys)):
        msg = f"camera_name_mapping creates duplicate output keys: {mapped_keys}"
        raise ValueError(msg)

    return dict(mapping)


def remap_camera_images(
    images: dict[str, Any],
    mapping: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Return image outputs renamed according to a validated camera mapping."""
    mapping = validate_camera_name_mapping(mapping, list(images))
    return {mapping.get(key, key): image for key, image in images.items()}


__all__ = ["remap_camera_images", "validate_camera_name_mapping"]
