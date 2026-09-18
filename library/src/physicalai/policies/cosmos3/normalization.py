# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action normalization for Cosmos 3, mirroring cosmos-framework.

A single affine seam ``(offset, scale)`` expresses every normalization method
cosmos-framework's ``resolve_action_normalization`` supports, so a checkpoint
trained here uses the same forward/inverse transform:

    normalize(x)   = (x - offset) / scale
    denormalize(y) =  y * scale   + offset

Methods:
    * ``none``         : identity (``offset=0``, ``scale=1``). Raw actions.
    * ``minmax``       : ``lo=min``, ``hi=max``  -> maps ``[min, max]`` to ``[-1, 1]``.
    * ``quantile``     : ``lo=q01``, ``hi=q99``.
    * ``quantile_rot`` : ``quantile`` on the ``global_raw`` (un-orthonormalized) stats block.
    * ``meanstd``      : ``offset=mean``, ``scale=std``.

Stats come from either an explicit cosmos-format JSON file (see
:func:`load_stats_file`) or the dataset's own statistics; the resolver is
agnostic to their origin.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

NormalizationMethod = str  # one of _METHODS
_METHODS: tuple[str, ...] = ("none", "minmax", "quantile", "quantile_rot", "meanstd")
_EPS = 1e-8


def _required_keys(method: NormalizationMethod) -> tuple[str, ...]:
    if method == "meanstd":
        return ("mean", "std")
    if method == "minmax":
        return ("min", "max")
    if method in {"quantile", "quantile_rot"}:
        return ("q01", "q99")
    return ()


def resolve_affine(
    method: NormalizationMethod,
    stats: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve a normalization method + stats into ``(offset, scale)`` tensors.

    Args:
        method: One of ``none``, ``minmax``, ``quantile``, ``quantile_rot``, ``meanstd``.
        stats: Mapping with the per-channel tensors the method needs
            (``mean``/``std``, ``min``/``max``, or ``q01``/``q99``).

    Returns:
        ``(offset, scale)`` float32 tensors; ``scale`` is clamped to ``>= 1e-8``.

    Raises:
        ValueError: If the method is unknown.
        KeyError: If a required stats key is missing.
    """
    if method not in _METHODS:
        msg = f"Unknown normalization method: {method!r}. Must be one of {_METHODS}."
        raise ValueError(msg)

    if method == "none":
        return torch.tensor(0.0), torch.tensor(1.0)

    missing = [k for k in _required_keys(method) if k not in stats]
    if missing:
        msg = f"Normalization method {method!r} requires stats keys {missing} that are not present."
        raise KeyError(msg)

    if method == "meanstd":
        offset = stats["mean"].float()
        scale = stats["std"].float()
    elif method == "minmax":
        lo, hi = stats["min"].float(), stats["max"].float()
        offset = (hi + lo) / 2.0
        scale = (hi - lo) / 2.0
    else:  # quantile / quantile_rot
        lo, hi = stats["q01"].float(), stats["q99"].float()
        offset = (hi + lo) / 2.0
        scale = (hi - lo) / 2.0

    return offset, scale.clamp(min=_EPS)


def load_stats_file(
    path: str | Path,
    method: NormalizationMethod,
) -> dict[str, torch.Tensor]:
    """Load a cosmos-format normalizer-stats JSON into per-channel tensors.

    Supports both the flat layout (``{"q01": [...], "q99": [...]}``) and the
    nested layout (``{"global": {...}, "global_raw": {...}}``). ``quantile_rot``
    reads the ``global_raw`` block; every other method reads ``global`` (or the
    top level for flat files).

    Args:
        path: Path to the JSON stats file.
        method: Normalization method the stats will be resolved with; selects the
            nested stats block for ``quantile_rot``.

    Returns:
        Mapping of stat name (``mean``/``std``/``min``/``max``/``q01``/``q99``)
        to a 1-D float32 tensor.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the requested nested stats block is absent.
    """
    stats_path = Path(path)
    if not stats_path.is_file():
        msg = f"Normalizer stats file not found: {stats_path}"
        raise FileNotFoundError(msg)

    with stats_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    block: dict[str, object] = raw
    if "global" in raw or "global_raw" in raw:
        block_key = "global_raw" if method == "quantile_rot" else "global"
        if block_key not in raw:
            msg = f"Stats block {block_key!r} not found in {stats_path}."
            raise KeyError(msg)
        block = raw[block_key]

    stat_names = ("mean", "std", "min", "max", "q01", "q99")
    return {
        name: torch.tensor(block[name], dtype=torch.float32)
        for name in stat_names
        if name in block and isinstance(block[name], list)
    }
