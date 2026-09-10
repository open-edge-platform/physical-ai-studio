# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature utilities for policy preprocessing and postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.data import Feature, FeatureType


def get_feature_by_type(features: list[Feature], feature_type: FeatureType) -> Feature | None:
    """Return the first feature that matches a given feature type.

    Args:
        features: List of feature definitions.
        feature_type: The feature type to search for.

    Returns:
        Feature | None: The first matching feature, or None if not found.
    """
    for feature in features:
        if feature.ftype == feature_type:
            return feature
    return None
