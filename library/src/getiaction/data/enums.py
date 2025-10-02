# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enum types."""

from enum import Enum


class BatchObservationComponents(str, Enum):
    STATE = "state"
    ACTION = "action"
    IMAGES = "images"
    EXTRA = "extra"


class FeatureType(str, Enum):
    VISUAL = "VISUAL"
    ACTION = "ACTION"
    STATE = "STATE"
    ENV = "ENV"


class NormalizationType(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
