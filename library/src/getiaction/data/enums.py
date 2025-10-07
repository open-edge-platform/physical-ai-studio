# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enum types."""

from enum import StrEnum


class BatchObservationComponents(StrEnum):
    """Enum for batch observation components."""

    STATE = "state"
    ACTION = "action"
    IMAGES = "images"
    EXTRA = "extra"


class FeatureType(StrEnum):
    """Enum for feature types."""

    VISUAL = "VISUAL"
    ACTION = "ACTION"
    STATE = "STATE"
    ENV = "ENV"


class NormalizationType(StrEnum):
    """Enum for feature normalization methods."""

    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
