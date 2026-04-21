# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stats-based observation normalizer for exported model packages.

Loads dataset statistics from a `safetensors` file and normalizes
observation features before inference.  Designed for use with models
exported by LeRobot where normalization is *not* baked into the
model graph.

Supports two normalization modes:

- **mean_std**: ``(x - mean) / (std + eps)``
- **min_max**: ``2 * (x - min) / (max - min + eps) - 1`` → [-1, 1]
"""

from __future__ import annotations

from typing import override

import numpy as np

from physicalai.inference.preprocessors.base import Preprocessor

_EPS = 1e-8


class StatsNormalizer(Preprocessor):
    """Normalize observation features using dataset statistics.

    Statistics are loaded lazily on first call (or eagerly via
    :meth:`load_stats`) from a ``safetensors`` file whose keys follow
    the ``{feature}/{stat}`` convention (e.g.
    ``observation.state/mean``, ``observation.state/std``).

    Only keys listed in *features* are transformed; all other keys
    pass through unchanged.

    Args:
        stats_path: Path to the ``safetensors`` file containing
            normalization statistics.
        mode: Normalization mode — ``"mean_std"`` or ``"min_max"``.
            Defaults to ``"mean_std"``.
        features: Feature names to normalize.  When empty,
            **all** features found in the stats file are normalized.
        artifact: Alias for *stats_path* used by manifest artifact
            resolution.  Provide one or the other, not both.

    Examples:
        Constructed via manifest (type-based resolution)::

            {"type": "normalize", "mode": "mean_std",
             "artifact": "stats.safetensors",
             "features": ["observation.state"]}

        Constructed via manifest (class_path-based resolution)::

            {"class_path": "physicalai.inference.preprocessors.StatsNormalizer",
             "init_args": {"stats_path": "stats.safetensors",
                           "mode": "mean_std",
                           "features": ["observation.state"]}}
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
        """Initialize with stats file path and normalization config.

        Args:
            stats_path: Absolute or relative path to the safetensors
                stats file.  Mutually exclusive with *artifact*.
            mode: ``"mean_std"`` or ``"min_max"``.
            features: Feature names to normalize.  If ``None`` or empty,
                all features present in the stats file are normalized.
            artifact: Alias for *stats_path*, used when the path is
                supplied via manifest ``artifact`` resolution.
            stats: Optional pre-loaded stats dict.  If provided, skips
                lazy loading from file.

        Raises:
            ValueError: If *mode* is not a recognized normalization mode
                or neither *stats_path* nor *artifact* is provided.
        """
        valid_modes = {"mean_std", "min_max", "identity"}
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
        self._stats: dict[str, dict[str, np.ndarray]] | None = stats

    def load_stats(self) -> None:
        """Eagerly load stats from the safetensors file.

        Called automatically on first ``__call__`` if not already loaded.
        """
        from safetensors.numpy import load_file  # noqa: PLC0415

        raw = load_file(self._stats_path)
        self._stats = _parse_flat_stats(raw)

    @override
    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._stats is None:
            self.load_stats()

        assert self._stats is not None  # noqa: S101

        if self._mode == "identity":
            return inputs

        outputs = dict(inputs)
        target_keys = self._features or set(self._stats.keys())

        for key in target_keys:
            if key not in outputs or key not in self._stats:
                continue
            outputs[key] = _normalize(outputs[key], self._stats[key], self._mode)

        return outputs

    def __repr__(self) -> str:
        """Return string representation."""
        features = sorted(self._features) if self._features else "all"
        return f"{self.__class__.__name__}(stats_path={self._stats_path!r}, mode={self._mode!r}, features={features})"


def _parse_flat_stats(raw: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Group flat ``{feature}/{stat}`` keys into nested dicts.

    ``{"observation.state/mean": ..., "observation.state/std": ...}``
    becomes ``{"observation.state": {"mean": ..., "std": ...}}``.

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


def _normalize(
    tensor: np.ndarray,
    stats: dict[str, np.ndarray],
    mode: str,
) -> np.ndarray:
    """Apply forward normalization to a single tensor.

    Returns:
        Normalized array.
    """
    if mode == "mean_std":
        mean = stats["mean"]
        std = stats["std"]
        return (tensor - mean) / (std + np.array(_EPS))

    if mode == "min_max":
        min_val = stats["min"]
        max_val = stats["max"]
        denom = max_val - min_val + _EPS
        return 2.0 * (tensor - min_val) / denom - 1.0

    return tensor
