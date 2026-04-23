# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint converter between Lightning and LeRobot safetensors formats."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

import torch

from physicalai.export.mixin_policy import CONFIG_KEY, POLICY_NAME_KEY
from physicalai.policies.lerobot.policy import LeRobotPolicy

logger = logging.getLogger(__name__)

_LEROBOT_PREFIX = "_lerobot_policy."


def lightning_to_lerobot(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    map_location: torch.device | str | None = "cpu",
) -> Path:
    """Convert a Lightning checkpoint to LeRobot-compatible format.

    Reconstructs the wrapper (config, dataset stats, weights, **and the
    pre/post-processors**) via :meth:`LeRobotPolicy.load_from_checkpoint`
    and persists it through :meth:`LeRobotPolicy.save_pretrained`. Going
    through the wrapper — instead of hand-rolling ``config.json`` /
    ``model.safetensors`` here — guarantees the output directory contains
    every artefact LeRobot's loader expects (in particular the processor
    JSONs that the previous hand-rolled converter silently dropped).

    Args:
        checkpoint_path: Path to the Lightning ``.ckpt`` file.
        output_dir: Directory where the LeRobot-format artefacts will be
            written. Created if missing.
        map_location: Device mapping for ``torch.load``.

    Returns:
        Path to the output directory.

    Note:
        Propagates ``KeyError`` from
        :meth:`LeRobotPolicy.load_from_checkpoint` if the checkpoint is
        missing its embedded config, and ``RuntimeError`` from
        :meth:`LeRobotPolicy.save_pretrained` if the reconstructed wrapper
        has no processors (e.g. checkpoints predating processor
        persistence).
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    policy = LeRobotPolicy.load_from_checkpoint(checkpoint_path, map_location=map_location)
    policy.save_pretrained(output_dir)

    logger.info("Converted %s -> %s (LeRobot-format directory)", checkpoint_path, output_dir)
    return output_dir


def lerobot_to_lightning(
    lerobot_dir: str | Path,
    output_path: str | Path,
    *,
    policy_name: str | None = None,
) -> Path:
    """Convert LeRobot safetensors format to a Lightning checkpoint.

    Creates a ``.ckpt`` file loadable by
    ``LeRobotPolicy.load_from_checkpoint()`` from a directory containing
    ``config.json`` and ``model.safetensors``.

    Args:
        lerobot_dir: Directory containing ``config.json`` and
            ``model.safetensors``.
        output_path: Path for the output ``.ckpt`` file.
        policy_name: Policy type name (e.g. ``"act"``). If *None*, inferred
            from ``config.json["type"]``.

    Returns:
        Path to the created ``.ckpt`` file.

    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If ``policy_name`` cannot be determined.
    """
    from safetensors.torch import load_file  # noqa: PLC0415

    lerobot_dir = Path(lerobot_dir)
    output_path = Path(output_path)

    config_path = lerobot_dir / "config.json"
    weights_path = lerobot_dir / "model.safetensors"

    if not config_path.exists():
        msg = f"Missing config.json in {lerobot_dir}"
        raise FileNotFoundError(msg)
    if not weights_path.exists():
        msg = f"Missing model.safetensors in {lerobot_dir}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        config_dict = json.load(f)

    if policy_name is None:
        policy_name = config_dict.get("type")
        if policy_name is None:
            msg = "Cannot determine policy_name: not in config.json['type'] and not provided."
            raise ValueError(msg)

    lerobot_state = load_file(str(weights_path))
    lightning_state = _add_prefix(lerobot_state, _LEROBOT_PREFIX)

    checkpoint: dict = {
        "state_dict": lightning_state,
        CONFIG_KEY: config_dict,
        POLICY_NAME_KEY: policy_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save(checkpoint, str(output_path))  # nosec B614

    logger.info("Converted %s -> %s", lerobot_dir, output_path)
    return output_path


def _add_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return OrderedDict((prefix + key, value) for key, value in state_dict.items())
