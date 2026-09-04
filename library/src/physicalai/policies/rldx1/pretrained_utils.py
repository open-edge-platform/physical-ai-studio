# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Checkpoint and policy utility helpers for the RLDX-1 policy.

The RLDX-1 weights ship as sharded ``safetensors`` files alongside a
``config.json``. These helpers resolve a HuggingFace repo (or local directory)
to a local snapshot and load the merged state dict using the ``safetensors``
backend only — never ``torch.load`` / pickle (lib.security rule 8).

This module also hosts lightweight policy utilities shared by construction,
schema resolution, and export-graph preparation paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Only these files are pulled from a remote checkpoint repo (lib.security rule 8:
# safetensors allowlist; no pickled ``*.bin`` / ``*.pt`` weights).
ALLOW_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]


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


def extract_camera_names(
    processor_config_path: Path | None,
    embodiment_tag: str = "general_embodiment",
) -> list[str]:
    """Read camera view names for one embodiment from ``processor_config.json``.

    Robust to a missing/unreadable file or an ``embodiment_tag``/``video`` section
    absent from it: any of those return an empty list rather than raising. Note
    RLWRLD checkpoints never record pixel resolution here (or anywhere else in
    the repo) -- callers must still supply that separately.

    Args:
        processor_config_path: Path to the ``processor_config.json`` file, or ``None``.
        embodiment_tag: Key selecting the embodiment's modality config block.

    Returns:
        Ordered camera view names (``video.modality_keys`` for ``embodiment_tag``
        under ``processor_kwargs.modality_configs``), or an empty list when
        unavailable.
    """
    if processor_config_path is None or not processor_config_path.exists():
        logger.warning("No processor_config.json found; camera names must be supplied explicitly.")
        return []

    with processor_config_path.open(encoding="utf-8") as f:
        processor_config = json.load(f)

    video_config = (
        processor_config.get("processor_kwargs", {}).get("modality_configs", {}).get(embodiment_tag, {}).get("video")
    )
    if not video_config:
        logger.warning(
            "Embodiment tag %r has no video modality config in %s; camera names must be supplied explicitly.",
            embodiment_tag,
            processor_config_path,
        )
        return []

    return list(video_config.get("modality_keys", []))
