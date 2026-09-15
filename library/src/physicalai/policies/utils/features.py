# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for resolving a feature's shape from a partial stats dict."""

from __future__ import annotations

from typing import Any

# Checked in order; the first stat vector present determines the inferred shape.
_SHAPE_INFERENCE_STAT_KEYS = ("mean", "std", "q01", "q99", "min", "max")


def infer_shape_from_stats(feature: dict[str, Any]) -> tuple[int, ...] | None:
    """Return a feature's shape, inferring it from a stat vector when absent.

    Some dataset-stats sources record an explicit ``"shape"`` key (e.g.
    LeRobot-style config-derived features); others only carry bare
    normalization vectors (e.g. RLDX1's ``extract_dataset_stats``). Callers
    decide their own fallback/error behavior when this returns ``None``.

    Returns:
        ``tuple(feature["shape"])`` if present, else ``(len(values),)`` for the
        first available stat vector, else ``None``.
    """
    if "shape" in feature:
        return tuple(feature["shape"])
    for stat_key in _SHAPE_INFERENCE_STAT_KEYS:
        values = feature.get(stat_key)
        if isinstance(values, (list, tuple)):
            return (len(values),)
    return None
