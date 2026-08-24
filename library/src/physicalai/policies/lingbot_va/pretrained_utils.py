# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loading LingBot-VA checkpoints published in the LeRobot layout.

A LingBot-VA checkpoint holds only the trainable ~5B transformer:

- ``config.json`` — the architecture and inference hyperparameters;
- ``model.safetensors`` — the transformer weights, already keyed ``transformer.*``;
- ``policy_postprocessor.json`` plus its safetensors state file — the per-channel action
  q01/q99 used to map predictions back to physical units.

The frozen VAE + UMT5 encoder + tokenizer (~20 GB) are *not* in the checkpoint; they are
pulled separately from ``config.wan_pretrained_path`` when the model first runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from physicalai.data.observation import ACTION

from .config import LingBotVAConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_HUB_KWARGS = frozenset({
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "token",
    "revision",
    "local_files_only",
})

_LEROBOT_ONLY_KEYS = frozenset({
    "type",
    "input_features",
    "output_features",
    "device",
    "use_amp",
    "use_peft",
    "push_to_hub",
    "repo_id",
    "private",
    "tags",
    "license",
    "pretrained_path",
    "normalization_mapping",
})


@dataclass(frozen=True)
class CheckpointFiles:
    """Resolved paths of a LingBot-VA checkpoint.

    Attributes:
        config_file: Path to ``config.json``.
        weights_file: Path to ``model.safetensors``.
        postprocessor_file: Path to ``policy_postprocessor.json``, if published.
        postprocessor_dir: Directory holding the postprocessor state files.
    """

    config_file: Path
    weights_file: Path
    postprocessor_file: Path | None = None
    postprocessor_dir: Path | None = None


def resolve_checkpoint(pretrained_name_or_path: str | Path, **kwargs: Any) -> CheckpointFiles:  # noqa: ANN401
    """Resolve a checkpoint's files from a local directory or the HuggingFace Hub.

    Args:
        pretrained_name_or_path: HuggingFace repo id or local directory.
        **kwargs: Forwarded to ``huggingface_hub.hf_hub_download`` (only the recognized
            download options are passed through).

    Returns:
        The resolved :class:`CheckpointFiles`.
    """
    path = Path(pretrained_name_or_path)
    if path.is_dir():
        local_postprocessor = path / "policy_postprocessor.json"
        has_postprocessor = local_postprocessor.exists()
        return CheckpointFiles(
            config_file=path / "config.json",
            weights_file=path / "model.safetensors",
            postprocessor_file=local_postprocessor if has_postprocessor else None,
            postprocessor_dir=path if has_postprocessor else None,
        )

    hub_kwargs = {k: v for k, v in kwargs.items() if k in _HUB_KWARGS}
    repo_id = str(pretrained_name_or_path)
    config_file = Path(str(hf_hub_download(repo_id, "config.json", **hub_kwargs)))  # nosec B615
    weights_file = Path(str(hf_hub_download(repo_id, "model.safetensors", **hub_kwargs)))  # nosec B615

    postprocessor_file: Path | None = None
    postprocessor_dir: Path | None = None
    try:
        postprocessor_file = Path(str(hf_hub_download(repo_id, "policy_postprocessor.json", **hub_kwargs)))  # nosec B615
        postprocessor_dir = postprocessor_file.parent
        with postprocessor_file.open(encoding="utf-8") as f:
            postprocessor_data = json.load(f)
        for step in postprocessor_data.get("steps", []):
            state_file = step.get("state_file")
            if state_file:
                hf_hub_download(repo_id, state_file, **hub_kwargs)  # nosec B615
    except Exception:  # noqa: BLE001
        logger.debug("No policy_postprocessor.json published for %s; actions stay unnormalized.", repo_id)
        postprocessor_file = None
        postprocessor_dir = None

    return CheckpointFiles(
        config_file=config_file,
        weights_file=weights_file,
        postprocessor_file=postprocessor_file,
        postprocessor_dir=postprocessor_dir,
    )


def load_config(config_file: Path, overrides: dict[str, Any] | None = None) -> LingBotVAConfig:
    """Parse a checkpoint's ``config.json`` into a :class:`LingBotVAConfig`.

    LeRobot-only keys (feature specs, device, hub metadata) are dropped; everything else is
    coerced by the dataclass. Caller overrides are applied last so training-time settings
    such as ``attn_mode`` or the optimizer always win over the published values.

    Args:
        config_file: Path to the checkpoint's ``config.json``.
        overrides: Values that take precedence over the file's contents. ``None`` entries
            are ignored, so callers can pass "unset" through.

    Returns:
        The resolved configuration.
    """
    with Path(config_file).open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    output_features = raw.get("output_features") or {}
    action_shape = (output_features.get(ACTION) or {}).get("shape")

    config_kwargs = {k: v for k, v in raw.items() if k not in _LEROBOT_ONLY_KEYS}
    if "used_action_channel_ids" not in config_kwargs and action_shape:
        config_kwargs["used_action_channel_ids"] = list(range(int(action_shape[0])))

    config_kwargs.update({key: value for key, value in (overrides or {}).items() if value is not None})

    # strict=False: tolerate keys published by newer/older LeRobot configs.
    return LingBotVAConfig.from_dict(config_kwargs, strict=False)


def extract_action_stats(
    postprocessor_file: Path | None,
    postprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Read the action normalization statistics out of a checkpoint's postprocessor.

    Args:
        postprocessor_file: Path to ``policy_postprocessor.json``, or ``None``.
        postprocessor_dir: Directory holding the referenced state files.

    Returns:
        Statistics keyed by feature name (``{"action": {"q01": ..., "q99": ...}}``), or an
        empty dict when the checkpoint publishes none.
    """
    if postprocessor_file is None or postprocessor_dir is None or not Path(postprocessor_file).exists():
        return {}

    with Path(postprocessor_file).open(encoding="utf-8") as f:
        config = json.load(f)

    stats: dict[str, dict[str, Any]] = {}
    for step in config.get("steps", []):
        registry_name = step.get("registry_name", "")
        state_file = step.get("state_file")
        if "normalizer" not in registry_name.lower() or not state_file:
            continue

        state_path = Path(postprocessor_dir) / state_file
        if not state_path.exists():
            logger.warning("Normalizer state file not found: %s", state_path)
            continue

        features = step.get("config", {}).get("features", {})
        for flat_key, tensor in load_file(str(state_path)).items():
            feature_name, stat_name = flat_key.rsplit(".", 1)
            entry = stats.setdefault(
                feature_name,
                {"name": feature_name, "shape": tuple(features.get(feature_name, {}).get("shape", ()))},
            )
            entry[stat_name] = tensor.cpu().tolist()

    return stats


def detect_normalization_mode(postprocessor_file: Path | None) -> str | None:
    """Detect the action normalization mode a checkpoint was published with.

    Args:
        postprocessor_file: Path to ``policy_postprocessor.json``, or ``None``.

    Returns:
        ``"QUANTILES"``, ``"MEAN_STD"``, or ``None`` when it cannot be determined.
    """
    if postprocessor_file is None or not Path(postprocessor_file).exists():
        return None

    with Path(postprocessor_file).open(encoding="utf-8") as f:
        config = json.load(f)

    modes = {
        value
        for step in config.get("steps", [])
        for value in (step.get("config", {}).get("norm_map") or {}).values()
        if value != "IDENTITY"
    }
    if modes == {"QUANTILES"}:
        return "QUANTILES"
    if modes == {"MEAN_STD"}:
        return "MEAN_STD"
    return None


def load_transformer_weights(model: torch.nn.Module, weights_file: Path) -> None:
    """Load a checkpoint's ``transformer.*`` weights into a :class:`LingBotVAModel`.

    The published state dict is already keyed for the model, so no remapping is needed;
    ``strict=False`` only tolerates the frozen sub-models being absent.

    Args:
        model: The model to load into.
        weights_file: Path to ``model.safetensors``.
    """
    state_dict = load_file(str(weights_file))
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        logger.warning("Missing keys when loading LingBot-VA weights: %d (e.g. %s)", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading LingBot-VA weights: %d (e.g. %s)", len(unexpected), unexpected[:5])


__all__ = [
    "CheckpointFiles",
    "detect_normalization_mode",
    "extract_action_stats",
    "load_config",
    "load_transformer_weights",
    "resolve_checkpoint",
]
