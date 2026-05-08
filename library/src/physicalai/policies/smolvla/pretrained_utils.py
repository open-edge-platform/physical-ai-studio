# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utilities for loading pretrained SmolVLA weights and dataset stats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.data.observation import ACTION, STATE, FeatureType
from physicalai.policies.pi05.pretrained_utils import extract_dataset_stats as pi05_extract_dataset_stats

if TYPE_CHECKING:
    from pathlib import Path


def fix_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Fix state dict keys from pretrained SmolVLA to match model keys.

    Returns:
        State dict with ``model.`` prefix renamed to ``_model.``.
    """
    return {
        key.replace("model.", "_model.", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }


def parse_config_features(hf_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build stats from config.json ``input_features``/``output_features``.

    When no preprocessor stats file is available, build identity-like stats
    from feature shapes in the config.

    Returns:
        Dict mapping feature names to stat dicts.
    """
    stats: dict[str, dict[str, Any]] = {}

    for section in ("input_features", "output_features"):
        features = hf_config.get(section, {})
        if not isinstance(features, dict):
            continue
        for feat_name, feat_info in features.items():
            if not isinstance(feat_info, dict):
                continue
            shape = feat_info.get("shape", None)
            if shape is None:
                continue
            shape = tuple(shape)
            dim = shape[0] if shape else 1
            f_type = feat_info.get("type", "UNKNOWN")

            if STATE in feat_name.lower() or ACTION in feat_name.lower() or f_type == FeatureType.VISUAL.value:
                feature_alias = feat_name
                feature_alias = feature_alias.removeprefix("observation.")
                stats[feat_name] = {
                    "name": feature_alias,
                    "shape": shape,
                    "type": f_type,
                    "mean": [0.0] * dim,
                    "std": [1.0] * dim,
                }

    return stats


def extract_dataset_stats(
    hf_config: dict[str, Any],
    preprocessor_file: Path | None,
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Build ``dataset_stats`` dict that ``make_pi05_preprocessors`` expects.

    The stats format expected by the preprocessor is:
    ``{feature_name: {"name": str, "shape": tuple, ...stat fields...}}``
    where stat fields include whichever of mean/std, q01/q99, or min/max
    are present in the pretrained model's artifacts.

    The normalization mode (detected from the preprocessor JSON) determines
    which stat fields the normalizer actually uses at runtime.

    Returns:
        Dataset stats dict mapping feature names to stat dicts.
    """
    # this gives a skeleton of the input / output features, but w/o normalization data
    config_features = parse_config_features(hf_config)

    # extra info with normalization
    processing_stats = pi05_extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

    for f_name in config_features:
        if STATE in f_name.lower():
            for proc_f_name in processing_stats:
                if STATE in proc_f_name.lower():
                    config_features[f_name]["mean"] = processing_stats[proc_f_name]["mean"]
                    config_features[f_name]["std"] = processing_stats[proc_f_name]["std"]
        elif ACTION in f_name.lower():
            for proc_f_name in processing_stats:
                if ACTION in proc_f_name.lower():
                    config_features[f_name]["mean"] = processing_stats[proc_f_name]["mean"]
                    config_features[f_name]["std"] = processing_stats[proc_f_name]["std"]

    return config_features
