# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compute quantile statistics for LeRobot datasets that lack them.

Older LeRobot datasets (pre-quantile era) only store mean/std/min/max.
This module delegates to LeRobot's own statistics computation so that
the resulting q01/q99 values are identical to those produced by
``lerobot.scripts.augment_dataset_quantile_stats``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def has_quantile_stats(dataset: LeRobotDataset) -> bool:
    """Check whether the dataset's pre-computed stats already contain q01/q99."""
    for feature_stats in dataset.meta.stats.values():
        if "q01" in feature_stats or "q99" in feature_stats:
            return True
    return False


def augment_dataset_quantile_stats(dataset: LeRobotDataset) -> None:
    """Compute q01/q99 quantile stats in-place for a ``LeRobotDataset``.

    Delegates to LeRobot's ``compute_quantile_stats_for_dataset`` so that
    the computed values match those produced by the upstream
    ``augment_dataset_quantile_stats`` script.

    Only injects the ``q01`` and ``q99`` keys into ``dataset.meta.stats``;
    existing keys (mean, std, min, max) are left untouched.

    Args:
        dataset: A ``LeRobotDataset`` instance.  Its ``meta.stats`` is
            modified in-place.
    """
    from lerobot.scripts.augment_dataset_quantile_stats import (  # noqa: PLC0415
        compute_quantile_stats_for_dataset,
    )

    logger.info(
        "Computing quantile stats via LeRobot for %d episodes",
        dataset.num_episodes,
    )

    new_stats = compute_quantile_stats_for_dataset(dataset)

    # Inject only q01/q99 into existing meta.stats
    for key, feat_stats in new_stats.items():
        if key not in dataset.meta.stats:
            continue
        for q_key in ("q01", "q99"):
            if q_key in feat_stats:
                val = feat_stats[q_key]
                if not isinstance(val, torch.Tensor):
                    val = torch.from_numpy(val).float()
                dataset.meta.stats[key][q_key] = val

    logger.info("Quantile stats computed via LeRobot for dataset")
