# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained Pi05 weights from HuggingFace/lerobot format.

Handles:
- Normalization stat conversion (QUANTILES/MIN_MAX → MEAN_STD)
- State dict key remapping (lerobot → Pi05Model)
- Feature shape resolution from config or stats
"""

from __future__ import annotations

import json
import logging
import re
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
    """Build ``dataset_stats`` dict that ``make_pi05_preprocessors`` expects.

    The stats format expected by the preprocessor is:
    ``{feature_name: {"name": str, "shape": tuple, "mean": list, "std": list}}``

    Lerobot models use QUANTILES normalization with q01/q99 stats.
    We convert to mean/std that produces an equivalent [-1, 1] mapping:
    ``mean = (q01 + q99) / 2``  and  ``std = (q99 - q01) / 2``.

    Returns:
        Dataset stats dict mapping feature names to stat dicts.
    """
    stats: dict[str, dict[str, Any]] = {}

    # Try to extract stats from preprocessor config + state file
    if preprocessor_file is not None and preprocessor_file.exists():
        try:
            with preprocessor_file.open(encoding="utf-8") as f:
                preproc_config = json.load(f)
            stats = parse_preprocessor_stats(preproc_config, hf_config, preprocessor_dir)
            if stats:
                return stats
        except Exception:  # noqa: BLE001
            msg = "Could not parse preprocessor file, falling back to config.json"
            logger.debug(msg)

    # Fallback: build minimal stats from config.json features
    return parse_config_features(hf_config)


def parse_preprocessor_stats(
    preproc_config: dict[str, Any],
    hf_config: dict[str, Any],
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Extract normalization stats from lerobot's policy_preprocessor.json.

    Lerobot stores the stats in a separate safetensors file (referenced
    by ``state_file`` in each pipeline step). The keys are flat:
    ``"observation.state.q01"``, ``"action.q99"``, etc.

    Returns:
        Dict mapping feature names to stat dicts with mean/std.
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
            msg = f"Normalizer state file not found: {state_path}"
            logger.warning(msg)
            continue

        # Load the flat tensor dict: {"observation.state.q01": tensor, ...}
        tensor_stats = load_file(str(state_path))

        # Group by feature name: "observation.state.q01" → key="observation.state", stat="q01"
        grouped: dict[str, dict[str, list[float]]] = {}
        for flat_key, tensor in tensor_stats.items():
            feat_name, stat_name = flat_key.rsplit(".", 1)
            grouped.setdefault(feat_name, {})[stat_name] = tensor.cpu().tolist()

        # Convert each feature's stats to mean/std
        for feat_name, feat_stats in grouped.items():
            shape = resolve_feature_shape(feat_name, hf_config, feat_stats)
            mean, std = convert_normalization_stats(feat_stats)
            if mean is not None and std is not None:
                stats[feat_name] = {
                    "name": feat_name,
                    "shape": shape,
                    "mean": mean,
                    "std": std,
                }

    return stats


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

            if "state" in feat_name.lower() or feat_name == ACTION or "action" in feat_name.lower():
                stats[feat_name] = {
                    "name": feat_name,
                    "shape": shape,
                    "mean": [0.0] * dim,
                    "std": [1.0] * dim,
                }

    return stats


def resolve_feature_shape(feat_name: str, hf_config: dict[str, Any], feat_stats: dict[str, Any]) -> tuple[int, ...]:
    """Resolve shape for a feature, checking config features then stat tensors.

    Returns:
        Tuple representing the feature shape.
    """
    # Check config features
    for section in ("input_features", "output_features"):
        features = hf_config.get(section, {})
        if isinstance(features, dict) and feat_name in features:
            feat_info = features[feat_name]
            if isinstance(feat_info, dict) and "shape" in feat_info:
                return tuple(feat_info["shape"])

    # Infer from stats tensor lengths
    for key in ("mean", "std", "q01", "q99", "min", "max"):
        val = feat_stats.get(key)
        if isinstance(val, list):
            return (len(val),)

    return (1,)


def convert_normalization_stats(feat_stats: dict[str, Any]) -> tuple[list[float] | None, list[float] | None]:
    """Convert lerobot normalization stats to mean/std.

    Handles QUANTILES (q01/q99), MIN_MAX (min/max), and MEAN_STD (mean/std).
    QUANTILES conversion: mean = (q01+q99)/2, std = (q99-q01)/2.
    MIN_MAX conversion:   mean = (min+max)/2, std = (max-min)/2.

    Returns:
        Tuple of (mean, std) lists, or (None, None) if conversion fails.
    """
    # Already has mean/std
    if "mean" in feat_stats and "std" in feat_stats:
        mean = feat_stats["mean"]
        std = feat_stats["std"]
        if isinstance(mean, list) and isinstance(std, list):
            return mean, std

    # QUANTILES: q01/q99 → mean/std
    if "q01" in feat_stats and "q99" in feat_stats:
        q01 = feat_stats["q01"]
        q99 = feat_stats["q99"]
        if isinstance(q01, list) and isinstance(q99, list):
            mean = [(a + b) / 2.0 for a, b in zip(q01, q99, strict=False)]
            std = [max((b - a) / 2.0, 1e-8) for a, b in zip(q01, q99, strict=False)]
            return mean, std

    # MIN_MAX: min/max → mean/std
    if "min" in feat_stats and "max" in feat_stats:
        mn = feat_stats["min"]
        mx = feat_stats["max"]
        if isinstance(mn, list) and isinstance(mx, list):
            mean = [(a + b) / 2.0 for a, b in zip(mn, mx, strict=False)]
            std = [max((b - a) / 2.0, 1e-8) for a, b in zip(mn, mx, strict=False)]
            return mean, std

    return None, None


def fix_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fix state dict keys to match Pi05Model architecture.

    Adapted from lerobot's ``Pi05Policy._fix_pytorch_state_dict_keys``.

    Returns:
        State dict with corrected key names.
    """
    fixed: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        # Strip "model." prefix — HF checkpoint wraps everything
        # under the policy's `self.model`, but we load into Pi05Model directly.
        new_key = new_key.removeprefix("model.")

        # Skip adaRMS mismatch keys for expert
        if re.match(
            r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
            r"(input_layernorm|post_attention_layernorm)\.weight$",
            new_key,
        ):
            # Pi05 expert uses adaRMS — skip plain layernorm weights
            continue

        if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight$", new_key):
            continue

        # Rename action_time_mlp_* → time_mlp_*
        if new_key.startswith("action_time_mlp_in."):
            new_key = new_key.replace("action_time_mlp_in.", "time_mlp_in.")
        elif new_key.startswith("action_time_mlp_out."):
            new_key = new_key.replace("action_time_mlp_out.", "time_mlp_out.")

        # Skip state_proj (not used in pi05)
        if new_key.startswith("state_proj."):
            continue

        # Copy lm_head → embed_tokens (weight tying)
        if new_key == "paligemma_with_expert.paligemma.lm_head.weight":
            tied_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            fixed[tied_key] = value.clone()

        fixed[new_key] = value

    return fixed
