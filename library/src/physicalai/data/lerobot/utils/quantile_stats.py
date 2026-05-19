# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compute quantile statistics for LeRobot datasets that lack them.

Older LeRobot datasets (pre-quantile era) only store mean/std/min/max.
This module delegates to LeRobot's own statistics computation so that
the resulting q01/q99 values are identical to those produced by
``lerobot.scripts.augment_dataset_quantile_stats``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


def _to_tensor(val: object) -> torch.Tensor:
    """Convert a value to a float tensor if it isn't one already.

    Returns:
        Float tensor representation of *val*.
    """
    if isinstance(val, torch.Tensor):
        return val
    return torch.from_numpy(val).float()  # type: ignore[arg-type]


def has_quantile_stats(dataset: LeRobotDataset) -> bool:
    """Check whether the dataset's pre-computed stats already contain q01/q99.

    Returns:
        True if any feature has q01 or q99 stats.
    """
    return any("q01" in feature_stats or "q99" in feature_stats for feature_stats in dataset.meta.stats.values())


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
    logger.info(
        "Computing quantile stats via LeRobot for %d episodes",
        dataset.num_episodes,
    )

    new_stats = compute_quantile_stats_for_dataset(dataset)

    # Inject quantile stats into existing meta.stats, and add full stats
    # for any keys that are newly computed (e.g. image observations that
    # older datasets omitted from their pre-computed stats).
    stats = cast("dict[str, dict[str, Any]]", dataset.meta.stats)
    for key, feat_stats in new_stats.items():
        if key not in stats:
            stats[key] = {stat_key: _to_tensor(v) for stat_key, v in feat_stats.items()}
        else:
            for q_key in ("q01", "q99"):
                if q_key in feat_stats:
                    stats[key][q_key] = _to_tensor(feat_stats[q_key])

    logger.info("Quantile stats computed via LeRobot for dataset")
