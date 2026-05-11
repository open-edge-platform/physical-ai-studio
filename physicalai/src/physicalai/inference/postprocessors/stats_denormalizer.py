# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stats-based action denormalizer for exported model packages.

Loads dataset statistics from a ``safetensors`` file and
denormalizes model outputs (typically actions) after inference.
The inverse of :class:`~physicalai.inference.preprocessors.StatsNormalizer`.

Supports two denormalization modes:

- **mean_std**: ``x * std + mean``
- **min_max**: ``(x + 1) / 2 * (max - min) + min``
- **quantiles**: ``(x + 1) / 2 * (q99 - q01) + q01``
"""

from __future__ import annotations

from typing import Any, override

import numpy as np

from physicalai.inference.postprocessors.base import Postprocessor

_EPS = 1e-8


class StatsDenormalizer(Postprocessor):
    """Denormalize model outputs using dataset statistics.

    Statistics are loaded lazily on first call (or eagerly via
    :meth:`load_stats`) from a ``safetensors`` file whose keys follow
    the ``{feature}/{stat}`` convention (e.g. ``action/mean``,
    ``action/std``).

    Only keys listed in *features* are transformed; all other keys
    pass through unchanged.

    Args:
        stats_path: Path to the ``safetensors`` file containing
            normalization statistics.
        mode: Denormalization mode — ``"mean_std"`` or ``"min_max"``.
            Defaults to ``"mean_std"``.
        features: Feature names to denormalize.  When empty,
            **all** features found in the stats file are denormalized.
        artifact: Alias for *stats_path* used by manifest artifact
            resolution.  Provide one or the other, not both.
        stats: Optional pre-loaded stats dict.  If provided, skips
            lazy loading from file.

    Examples:
        Constructed via manifest (type-based resolution)::

            {"type": "denormalize", "mode": "mean_std",
             "artifact": "stats.safetensors",
             "features": ["action"]}

        Constructed via manifest (class_path-based resolution)::

            {"class_path": "physicalai.inference.postprocessors.StatsDenormalizer",
             "init_args": {"stats_path": "stats.safetensors",
                           "mode": "mean_std",
                           "features": ["action"]}}
    """

    def __init__(
        self,
        stats_path: str | None = None,
        mode: str = "mean_std",
        features: list[str] | None = None,
        *,
        artifact: str | None = None,
        stats: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        """Initialize with stats file path and denormalization config.

        Args:
            stats_path: Absolute or relative path to the safetensors
                stats file.  Mutually exclusive with *artifact*.
            mode: ``"mean_std"`` or ``"min_max"``.
            features: Feature names to denormalize.  If ``None`` or
                empty, all features present in the stats file are
                denormalized.
            artifact: Alias for *stats_path*, used when the path is
                supplied via manifest ``artifact`` resolution.
            stats: Optional pre-loaded stats dict.  If provided, skips
                lazy loading from file.

        Raises:
            ValueError: If *mode* is not a recognized normalization mode
                or neither *stats_path* nor *artifact* is provided.
        """
        valid_modes = {"mean_std", "min_max", "quantiles", "identity"}
        if mode not in valid_modes:
            msg = f"Unknown normalization mode {mode!r}. Expected one of {sorted(valid_modes)}"
            raise ValueError(msg)

        resolved_path = stats_path or artifact
        if resolved_path is None and stats is None:
            msg = "Either stats_path, artifact, or stats must be provided"
            raise ValueError(msg)

        self._stats_path = resolved_path
        self._mode = mode
        self._features: set[str] = set(features) if features else set()
        self._stats: dict[str, dict[str, np.ndarray]] | None = self._enforce_numpy(stats) if stats else None

    @staticmethod
    def _enforce_numpy(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        """Convert every value in a nested stats dict to a numpy array.

        Args:
            stats: Nested dict mapping feature names to stat-name → value dicts.
                Values may be lists, scalars, or already numpy arrays.

        Returns:
            A copy of the dict with all leaf values as ``np.ndarray``.
        """
        return {
            feature: {stat: np.asarray(val) for stat, val in stat_dict.items()} for feature, stat_dict in stats.items()
        }

    def load_stats(self) -> None:
        """Eagerly load stats from the safetensors file.

        Called automatically on first ``__call__`` if not already loaded.
        """
        from safetensors.numpy import load_file  # noqa: PLC0415

        raw = load_file(self._stats_path)
        self._stats = _parse_flat_stats(raw)

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._stats is None:
            self.load_stats()

        assert self._stats is not None  # noqa: S101

        if self._mode == "identity":
            return outputs

        result = dict(outputs)
        target_keys = self._features or set(self._stats.keys())

        for key in target_keys:
            if key not in result or key not in self._stats:
                continue
            result[key] = _denormalize(result[key], self._stats[key], self._mode)

        return result

    def __repr__(self) -> str:
        """Return string representation."""
        features = sorted(self._features) if self._features else "all"
        return f"{self.__class__.__name__}(stats_path={self._stats_path!r}, mode={self._mode!r}, features={features})"


def _parse_flat_stats(raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Group flat ``{feature}/{stat}`` keys into nested dicts.

    ``{"action/mean": ..., "action/std": ...}``
    becomes ``{"action": {"mean": ..., "std": ...}}``.

    Returns:
        Nested dict mapping feature names to stat-name → array dicts.
    """
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for flat_key, array in raw.items():
        if "/" not in flat_key:
            continue
        feature, stat_name = flat_key.rsplit("/", maxsplit=1)
        grouped.setdefault(feature, {})[stat_name] = array
    return grouped


def _denormalize(
    tensor: np.ndarray,
    stats: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Apply inverse normalization to a single tensor.

    Returns:
        Denormalized array.
    """
    if mode == "mean_std":
        mean = stats["mean"]
        std = stats["std"]
        return tensor * std + mean

    if mode == "min_max":
        min_val = stats["min"]
        max_val = stats["max"]
        return (tensor + 1.0) / 2.0 * (max_val - min_val) + min_val

    if mode == "quantiles":
        q01 = stats["q01"]
        q99 = stats["q99"]
        denom = q99 - q01
        denom = np.where(denom == 0, _EPS, denom)
        return (tensor + 1.0) * denom / 2.0 + q01

    return tensor
