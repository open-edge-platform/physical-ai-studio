# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained EO-1 weights and dataset stats."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from physicalai.policies.pi05.pretrained_utils import extract_dataset_stats as pi05_extract_dataset_stats
from physicalai.policies.smolvla.pretrained_utils import parse_config_features

if TYPE_CHECKING:
    from pathlib import Path

    import torch

logger = logging.getLogger(__name__)

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")

# `Tensor.data_ptr()` is 0 for tensors with no storage (meta tensors, empty tensors); those must not
# be treated as sharing memory with one another.
_NO_STORAGE = 0


def fix_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Adapt a published EO-1 state dict to :class:`EO1Model`.

    LeRobot's ``EO1Policy`` stores the network under ``model.``; the Studio model *is* that network,
    so the prefix is stripped and the weights load directly into ``policy.model``.

    Args:
        state_dict: Raw state dict loaded from the checkpoint.

    Returns:
        The remapped state dict.
    """
    return {key.removeprefix("model."): value for key, value in state_dict.items()}


def drop_tied_missing_keys(
    missing: list[str],
    loaded: set[str],
    current: dict[str, torch.Tensor],
) -> list[str]:
    """Drop reported-missing keys that are tied to a tensor the checkpoint did provide.

    Qwen ties ``lm_head.weight`` to ``embed_tokens.weight``, and safetensors refuses to serialize
    shared storage, so every published checkpoint omits one of the pair. The model already ties them
    at construction, so loading the survivor fills both - reporting the other as missing is noise
    that would fire on every load.

    Sharing is detected by storage identity rather than by a hard-coded key list, so it holds for
    whatever the backbone happens to tie.

    Args:
        missing: Keys ``load_state_dict`` reported as missing.
        loaded: Keys the checkpoint actually provided.
        current: The model's own ``state_dict()``.

    Returns:
        The genuinely missing keys.
    """
    loaded_storage = {
        current[key].data_ptr() for key in loaded if key in current and current[key].data_ptr() != _NO_STORAGE
    }
    kept = [key for key in missing if key not in current or current[key].data_ptr() not in loaded_storage]
    if dropped := len(missing) - len(kept):
        logger.debug("Ignored %d missing key(s) tied to weights the checkpoint provided.", dropped)
    return kept


def extract_dataset_stats(
    hf_config: dict[str, Any],
    preprocessor_file: Path | None,
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Build the ``dataset_stats`` dict that :func:`make_eo1_preprocessors` expects.

    Same shape as SmolVLA's loader, but every stat field is carried over rather than mean/std alone,
    so a policy configured for MIN_MAX or quantile normalization still finds what it needs.

    Args:
        hf_config: Parsed ``config.json`` of the pretrained repo.
        preprocessor_file: Path to ``policy_preprocessor.json``, when available.
        preprocessor_dir: Directory holding the referenced normalizer state files.

    Returns:
        Stats dict mapping feature names to stat dicts.
    """
    config_features = parse_config_features(hf_config)
    processing_stats = pi05_extract_dataset_stats(hf_config, preprocessor_file, preprocessor_dir)

    def same_kind(feature_name: str, candidate_name: str) -> bool:
        lowered, candidate = feature_name.lower(), candidate_name.lower()
        return any(kind in lowered and kind in candidate for kind in ("state", "action"))

    def stat_vector_len(stat: dict[str, Any]) -> int | None:
        for key in STAT_KEYS:
            value = stat.get(key)
            if isinstance(value, list):
                return len(value)
        return None

    for f_name, feature in config_features.items():
        feature_shape = feature.get("shape")
        expected_dim = feature_shape[0] if isinstance(feature_shape, tuple) and feature_shape else None

        for proc_f_name, proc_stats in processing_stats.items():
            if not same_kind(f_name, proc_f_name):
                continue
            actual_dim = stat_vector_len(proc_stats)
            if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
                continue
            for stat_key in STAT_KEYS:
                if stat_key in proc_stats:
                    feature[stat_key] = proc_stats[stat_key]
            break

    return config_features
