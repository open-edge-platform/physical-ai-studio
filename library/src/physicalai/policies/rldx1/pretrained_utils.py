# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Checkpoint loading helpers for the RLDX-1 policy.

The RLDX-1 weights ship as sharded ``safetensors`` files alongside a
``config.json``. These helpers resolve a HuggingFace repo (or local directory)
to a local snapshot and load the merged state dict using the ``safetensors``
backend only — never ``torch.load`` / pickle (lib.security rule 8).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Only these files are pulled from a remote checkpoint repo (lib.security rule 8:
# safetensors allowlist; no pickled ``*.bin`` / ``*.pt`` weights).
ALLOW_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]

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


def resolve_checkpoint_dir(
    base_model_path: str,
    *,
    revision: str | None = None,
) -> Path:
    """Resolve a repo id or local path to a local directory of weight files.

    Args:
        base_model_path: HuggingFace repo id or a local directory path.
        revision: Pinned git commit SHA for remote repos (lib.security rule 9).
            A concrete SHA, never a mutable branch/tag.

    Returns:
        Path to a local directory containing the ``safetensors`` shards and
        ``config.json``.
    """
    local = Path(base_model_path)
    if local.is_dir():
        return local

    if revision is None:
        logger.warning(
            "Loading %s without a pinned revision; pass a commit SHA to "
            "guarantee reproducible, tamper-evident weights (lib.security rule 9).",
            base_model_path,
        )

    snapshot = snapshot_download(
        base_model_path,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(snapshot)


def retrieve_safetensors_shards(base_model_path: str, revision: str | None) -> list[Path]:
    """Resolve the checkpoint directory and list its ``safetensors`` shards.

    Args:
        base_model_path: HuggingFace repo id or local directory path.
        revision: Pinned git commit SHA for remote repos (lib.security rule 9).

    Returns:
        Sorted list of shard file paths.

    Raises:
        FileNotFoundError: If no ``safetensors`` shards are found.
    """
    ckpt_dir = resolve_checkpoint_dir(base_model_path, revision=revision)
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    if not shards:
        msg = f"No *.safetensors weight shards found under {ckpt_dir}"
        raise FileNotFoundError(msg)
    return shards


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
