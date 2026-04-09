# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained SmolVLA weights from HuggingFace/lerobot format."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from safetensors.torch import load_file

from physicalai.data.observation import ACTION

if TYPE_CHECKING:
    from pathlib import Path

    import torch

logger = logging.getLogger(__name__)


def extract_dataset_stats(
    hf_config: dict[str, Any],
    preprocessor_file: Path | None,
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Build ``dataset_stats`` dict from pretrained HF checkpoint files.

    Tries the preprocessor normalizer safetensors first, falls back to
    building identity stats from config.json feature shapes.

    Returns:
        Mapping of feature name to stat dict with mean/std arrays.
    """
    if preprocessor_file is not None and preprocessor_file.exists():
        try:
            with preprocessor_file.open(encoding="utf-8") as f:
                preproc_config = json.load(f)
            stats = _parse_preprocessor_stats(preproc_config, hf_config, preprocessor_dir)
            if stats:
                return stats
        except Exception:  # noqa: BLE001
            logger.debug("Could not parse preprocessor file, falling back to config.json")

    return _parse_config_features(hf_config)


def fix_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap ``model.`` prefix to ``_model.`` to match SmolVLAModel layout.

    LeRobot's SmolVLAPolicy stores the VLAFlowMatching as ``self.model``,
    while physical-ai-studio's SmolVLAModel stores it as ``self._model``.

    Returns:
        State dict with remapped keys.
    """
    return {
        key.replace("model.", "_model.", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }


def _parse_preprocessor_stats(
    preproc_config: dict[str, Any],
    hf_config: dict[str, Any],
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Extract mean/std normalization stats from the normalizer safetensors.

    Returns:
        Mapping of feature name to stat dict with mean/std arrays.
    """
    stats: dict[str, dict[str, Any]] = {}

    steps = preproc_config.get("steps", [])
    if isinstance(steps, dict):
        steps = list(steps.values())

    for step in steps:
        step_type = step.get("registry_name", step.get("type", step.get("class_name", "")))
        if "normalizer" not in step_type.lower():
            continue

        state_file = step.get("state_file")
        if not state_file or preprocessor_dir is None:
            continue

        state_path = preprocessor_dir / state_file
        if not state_path.exists():
            logger.warning("Normalizer state file not found: %s", state_path)
            continue

        tensor_stats = load_file(str(state_path))

        grouped: dict[str, dict[str, list[float]]] = {}
        for flat_key, tensor in tensor_stats.items():
            feat_name, stat_name = flat_key.rsplit(".", 1)
            grouped.setdefault(feat_name, {})[stat_name] = tensor.cpu().tolist()

        for feat_name, feat_stats in grouped.items():
            mean = feat_stats.get("mean")
            std = feat_stats.get("std")
            if not isinstance(mean, list) or not isinstance(std, list):
                continue
            shape = _resolve_feature_shape(feat_name, hf_config, feat_stats)
            stats[feat_name] = {
                "name": feat_name,
                "shape": shape,
                "mean": mean,
                "std": std,
            }

    return stats


def _parse_config_features(hf_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build identity stats from config.json feature shapes (fallback).

    Returns:
        Mapping of feature name to stat dict with zero-mean/unit-std arrays.
    """
    stats: dict[str, dict[str, Any]] = {}

    for section in ("input_features", "output_features"):
        features = hf_config.get(section, {})
        if not isinstance(features, dict):
            continue
        for feat_name, feat_info in features.items():
            if not isinstance(feat_info, dict):
                continue
            shape = feat_info.get("shape")
            if shape is None:
                continue
            shape = tuple(shape)
            dim = shape[0] if shape else 1

            if "state" in feat_name.lower() or feat_name == ACTION or "action" in feat_name.lower():
                stats[feat_name] = {
                    "name": feat_name,
                    "shape": shape,
                    "mean": [0.0] * dim,
                    "std": [1.0] * dim,
                }

    return stats


def _resolve_feature_shape(
    feat_name: str,
    hf_config: dict[str, Any],
    feat_stats: dict[str, Any],
) -> tuple[int, ...]:
    """Resolve shape from config features, falling back to stat tensor length.

    Returns:
        Feature shape tuple.
    """
    for section in ("input_features", "output_features"):
        features = hf_config.get(section, {})
        if isinstance(features, dict) and feat_name in features:
            feat_info = features[feat_name]
            if isinstance(feat_info, dict) and "shape" in feat_info:
                return tuple(feat_info["shape"])

    for key in ("mean", "std"):
        val = feat_stats.get(key)
        if isinstance(val, list):
            return (len(val),)

    return (1,)
