# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Dataset-stats and schema helpers for the RLDX-1 policy."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.utils.features import infer_shape_from_stats

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Neutral fallback stats, used whenever a joint or stat field is missing from
# the on-disk statistics: mean=0/std=1 is a no-op for MEAN_STD normalization,
# min=-1/max=1/q01=-1/q99=1 keeps MIN_MAX/QUANTILES normalization well-defined
# (see FeatureNormalizeTransform in physicalai.policies.utils.normalization).
_DEFAULT_STAT_VALUES: dict[str, float] = {
    "min": -1.0,
    "max": 1.0,
    "mean": 0.0,
    "std": 1.0,
    "q01": -1.0,
    "q99": 1.0,
}


def merge_explicit_features(
    dataset_stats: dict[str, dict[str, Any]] | None,
    input_features: dict[str, Feature] | None,
    output_features: dict[str, Feature] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge explicit ``Feature`` overrides into a ``dataset_stats``-shaped dict.

    User-supplied features take precedence over anything already in
    ``dataset_stats`` (e.g. auto-fetched state/action stats) -- required for
    RLWRLD checkpoints, whose ``statistics.json`` never records camera shapes.

    Returns:
        The merged dict, or ``None`` if there is nothing to merge.
    """
    merged = dict(dataset_stats or {})
    for name, feature in {**(input_features or {}), **(output_features or {})}.items():
        if feature.shape is None:
            continue
        if feature.ftype == FeatureType.VISUAL:
            key = f"observation.{IMAGES}.{name}"
        elif feature.ftype == FeatureType.STATE:
            key = f"observation.{STATE}"
        elif feature.ftype == FeatureType.ACTION:
            key = ACTION
        else:
            key = name
        merged[key] = {"name": feature.name or name, "shape": feature.shape, "type": str(feature.ftype)}
    return merged or None


def infer_num_views_from_stats(dataset_stats: dict[str, dict[str, Any]] | None) -> int | None:
    """Infer the number of visual views present in dataset stats.

    Counts visual entries under ``observation.images``. Returns ``None`` when
    no visual feature can be identified.

    Returns:
        Number of visual views, or ``None`` if no visual feature is present.
    """
    if not dataset_stats:
        return None

    visual_keys: set[str] = set()
    prefix = f"observation.{IMAGES}."
    root_key = f"observation.{IMAGES}"

    for key, feature in dataset_stats.items():
        feature_type = str(feature.get("type", "")).lower()
        if key.startswith(prefix):
            visual_keys.add(key)
            continue
        if key == root_key:
            visual_keys.add(key)
            continue
        if "visual" in feature_type:
            visual_keys.add(key)

    if not visual_keys:
        return None

    if root_key in visual_keys:
        named_view_count = sum(1 for key in visual_keys if key.startswith(prefix))
        return named_view_count or 1

    return len(visual_keys)


def resolve_feature_shape(feature: dict[str, Any]) -> tuple[int, ...]:
    """Return a feature's shape, raising if it can't be inferred from stats.

    RLDX1's own ``extract_dataset_stats`` (used when loading a raw HF release
    checkpoint via ``_from_hf``) returns bare ``min``/``max``/``mean``/``std``/
    ``q01``/``q99`` vectors with no ``"shape"`` key at all -- unlike the
    LeRobot-style enriched stats (e.g. a Studio-trained checkpoint's full
    ``train_dataset.stats``, or an explicit ``input_features``/``output_features``
    override) which carry an explicit ``"shape"``. Both are valid dataset_stats
    entries for this policy; :func:`infer_shape_from_stats` handles both.

    Returns:
        The feature's shape as a tuple.

    Raises:
        ValueError: If neither ``"shape"`` nor a stat vector is present.
    """
    shape = infer_shape_from_stats(feature)
    if shape is None:
        msg = f"Cannot resolve a shape for feature {feature!r}: no 'shape' key and no stat vector to infer it from."
        raise ValueError(msg)
    return shape


def get_dataset_stats_entry(dataset_stats: dict[str, dict[str, Any]], *keys: str) -> dict[str, Any]:
    """Return the first present entry among candidate ``dataset_stats`` keys.

    ``extract_dataset_stats`` (raw HF release checkpoints) uses bare keys like
    ``"state"``; a Studio-trained checkpoint's full LeRobot-style stats use
    ``"observation.state"``. Callers pass both spellings as candidates.

    Returns:
        The matching stats dict.

    Raises:
        KeyError: If none of ``keys`` is present in ``dataset_stats``.
    """
    for key in keys:
        if key in dataset_stats:
            return dataset_stats[key]
    msg = f"None of {keys!r} found in dataset_stats (keys present: {sorted(dataset_stats)!r})"
    raise KeyError(msg)


def extract_dataset_stats(
    stats_path: Path | None,
    embodiment_tag: str = "general_embodiment",
    max_state_dim: int = 64,
    max_action_dim: int = 64,
) -> dict[str, dict[str, Any]]:
    """Build ``{"state": {...}, "action": {...}}`` normalization stats.

    Robust to a missing/unreadable stats file, an ``embodiment_tag`` absent
    from the file, or a missing ``state``/``action`` section: any of those
    fall back to neutral stats (see ``_DEFAULT_STAT_VALUES``) padded to
    ``max_state_dim`` / ``max_action_dim`` so model construction never fails
    for lack of dataset statistics.

    Args:
        stats_path: Path to the ``statistics.json`` file, or ``None``.
        embodiment_tag: Key selecting the embodiment's stats block.
        max_state_dim: Fallback vector length for the ``state`` section.
        max_action_dim: Fallback vector length for the ``action`` section.

    Returns:
        Dict with ``"state"`` and ``"action"`` keys, each mapping
        ``min``/``max``/``mean``/``std``/``q01``/``q99`` to flat float lists.
    """
    state_keys = ("min", "max", "mean", "std", "q01", "q99")

    stats: dict[str, Any] = {}
    if stats_path is None:
        logger.warning("No dataset stats path provided; using default normalization stats.")
    elif not stats_path.exists():
        logger.warning("Dataset stats file %s not found; using default normalization stats.", stats_path)
    else:
        with stats_path.open(encoding="utf-8") as f:
            stats = json.load(f)

    embodiment_stats = stats.get(embodiment_tag)
    if embodiment_stats is None:
        logger.warning(
            "Embodiment tag %r not found in dataset stats%s; using default normalization stats.",
            embodiment_tag,
            f" ({stats_path})" if stats_path is not None else "",
        )
        embodiment_stats = {}

    def _concat(
        section_name: str,
        section: dict | None,
        dim: int,
    ) -> dict[str, list[float]]:
        if not section:
            logger.warning(
                "No %r stats for embodiment %r; filling %d-dim defaults.",
                section_name,
                embodiment_tag,
                dim,
            )
            return {stat_key: [_DEFAULT_STAT_VALUES[stat_key]] * dim for stat_key in state_keys}

        out: dict[str, list[float]] = {}
        order = []
        for joint, joint_stats in section.items():
            order.append(joint)
            joint_dim = len(next(iter(joint_stats.values())))
            for stat_key in state_keys:
                values = joint_stats.get(stat_key)
                if values is None:
                    values = [_DEFAULT_STAT_VALUES[stat_key]] * joint_dim
                out.setdefault(stat_key, []).extend(values)
        logger.debug("%s.%s fields: %s", embodiment_tag, section_name, order)
        return out

    return {
        "action": _concat("action", embodiment_stats.get("action"), max_action_dim),
        "state": _concat("state", embodiment_stats.get("state"), max_state_dim),
    }
