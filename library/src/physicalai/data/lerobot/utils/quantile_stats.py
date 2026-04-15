# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compute quantile statistics for LeRobot datasets that lack them.

Older LeRobot datasets (pre-quantile era) only store mean/std/min/max.
This module computes q01 and q99 over the full dataset using a streaming
histogram approach (adapted from LeRobot's ``RunningQuantileStats``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

QUANTILES = (0.01, 0.99)


def has_quantile_stats(dataset: LeRobotDataset) -> bool:
    """Check whether the dataset's pre-computed stats already contain q01/q99."""
    for feature_stats in dataset.meta.stats.values():
        if "q01" in feature_stats or "q99" in feature_stats:
            return True
    return False


def augment_dataset_quantile_stats(dataset: LeRobotDataset) -> None:
    """Compute q01/q99 quantile stats in-place for a ``LeRobotDataset``.

    Iterates over all samples, collects state/action tensors, and computes
    per-feature q01 and q99 using ``numpy.quantile``. The results are
    injected into ``dataset.meta.stats`` so downstream consumers (e.g.
    ``_LeRobotDatasetAdapter``) pick them up automatically.

    Only processes non-image, non-string features that have existing stats
    entries (observation.state, action, etc.).

    Args:
        dataset: A ``LeRobotDataset`` instance.  Its ``meta.stats`` is
            modified in-place.
    """
    features_to_compute: list[str] = []
    for key in dataset.meta.stats:
        feat_info = dataset.features.get(key, {})
        dtype = feat_info.get("dtype", "")
        if dtype in ("image", "video", "string"):
            continue
        if key.startswith(("observation.", "action")):
            features_to_compute.append(key)

    if not features_to_compute:
        return

    logger.info(
        "Computing quantile stats (q01/q99) for %d features over %d samples",
        len(features_to_compute),
        len(dataset),
    )

    # Use sub-sampling for large datasets to avoid excessive iteration time.
    n = len(dataset)
    max_samples = 10_000
    if n > max_samples:
        indices = np.round(np.linspace(0, n - 1, max_samples)).astype(int).tolist()
        logger.info("Sub-sampling %d of %d frames for quantile estimation", max_samples, n)
    else:
        indices = list(range(n))

    # Collect data per feature
    collected: dict[str, list[np.ndarray]] = {k: [] for k in features_to_compute}

    for idx in indices:
        item = dataset[idx]
        for key in features_to_compute:
            val = item.get(key)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            collected[key].append(val)

    # Compute quantiles and inject into meta.stats
    for key, arrays in collected.items():
        if not arrays:
            continue
        stacked = np.stack(arrays, axis=0).astype(np.float32)

        # stacked shape: (N, *feature_shape)
        # Collapse all but the last dim for per-feature quantiles
        flat = stacked.reshape(-1, stacked.shape[-1])

        q01 = np.quantile(flat, 0.01, axis=0)
        q99 = np.quantile(flat, 0.99, axis=0)

        # Store as torch tensors to match existing stats format
        dataset.meta.stats[key]["q01"] = torch.from_numpy(q01).float()
        dataset.meta.stats[key]["q99"] = torch.from_numpy(q99).float()

    logger.info("Quantile stats computed for features: %s", features_to_compute)
