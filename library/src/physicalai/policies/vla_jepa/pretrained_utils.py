# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for loading pretrained VLA-JEPA weights and dataset stats."""

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
WORLD_MODEL_PREFIXES = ("video_encoder.", "video_predictor.")

# Modules that published checkpoints carry but this port deliberately does not build: upstream never
# calls `state_encoder`, and `extrinsics_encoder` only exists when the predictor consumes camera
# extrinsics. They are dropped rather than reported as unexpected keys on load.
UNUSED_MODULE_PREFIXES = ("video_predictor.state_encoder.", "video_predictor.extrinsics_encoder.")


def fix_state_dict_keys(state_dict: dict[str, Any], *, enable_world_model: bool = True) -> dict[str, Any]:
    """Adapt a published VLA-JEPA state dict to :class:`VLAJEPAModel`.

    LeRobot's ``VLAJEPAPolicy`` stores the network under ``model.``; the Studio model *is* that
    network, so the prefix is stripped and the weights load directly into ``policy.model``.

    Args:
        state_dict: Raw state dict loaded from the checkpoint.
        enable_world_model: When False, world-model tensors are dropped instead of being reported
            as unexpected keys.

    Returns:
        The remapped state dict.
    """
    fixed = {key.removeprefix("model."): value for key, value in state_dict.items()}
    if not enable_world_model:
        dropped = [key for key in fixed if key.startswith(WORLD_MODEL_PREFIXES)]
        for key in dropped:
            del fixed[key]
        if dropped:
            logger.info("Dropped %d world-model tensor(s): enable_world_model is False.", len(dropped))
    return fixed


def drop_unused_module_keys(state_dict: dict[str, Any], current: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Drop checkpoint tensors belonging to modules this port does not build.

    Only keys the model has no slot for are dropped, so a configuration that *does* build one of
    :data:`UNUSED_MODULE_PREFIXES` still receives its pretrained weights.

    Args:
        state_dict: Checkpoint tensors, already remapped by :func:`fix_state_dict_keys`.
        current: The model's own ``state_dict()``.

    Returns:
        The state dict without the unused tensors.
    """
    dropped = [key for key in state_dict if key.startswith(UNUSED_MODULE_PREFIXES) and key not in current]
    if dropped:
        logger.debug("Dropped %d checkpoint tensor(s) for modules this port does not build.", len(dropped))
    return {key: value for key, value in state_dict.items() if key not in set(dropped)}


def filter_reinit_modules(
    state_dict: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    prefixes: list[str] | None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop checkpoint tensors whose shape mismatches, when their prefix allows it.

    This is what makes cross-embodiment transfer work: when fine-tuning a pretrained model on a
    robot with a different action or state dimensionality, the input/output projections must be
    re-initialised from scratch while the rest of the network keeps its pretrained weights.

    Args:
        state_dict: Checkpoint tensors, already remapped by :func:`fix_state_dict_keys`.
        current: The model's own ``state_dict()``.
        prefixes: Key prefixes allowed to mismatch. None or empty means no mismatch is tolerated.

    Returns:
        Tuple of (tensors safe to load, human-readable descriptions of what was re-initialised).

    Raises:
        ValueError: If a tensor's shape mismatches and its prefix is not allowed to.
    """
    allowed = [p.removeprefix("model.") for p in (prefixes or [])]
    reinitialized: list[str] = []
    filtered: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if key in current and value.shape != current[key].shape:
            if not any(key.startswith(prefix) for prefix in allowed):
                msg = (
                    f"Shape mismatch for '{key}' (checkpoint {tuple(value.shape)} vs model "
                    f"{tuple(current[key].shape)}) and its prefix is not in `reinit_modules`."
                )
                raise ValueError(msg)
            reinitialized.append(f"{key}: checkpoint {tuple(value.shape)} -> model {tuple(current[key].shape)}")
        else:
            filtered[key] = value

    if reinitialized:
        logger.warning(
            "reinit_modules: skipping %d tensor(s) with mismatched shapes (randomly re-initialised):\n  %s",
            len(reinitialized),
            "\n  ".join(reinitialized),
        )
    return filtered, reinitialized


def extract_dataset_stats(
    hf_config: dict[str, Any],
    preprocessor_file: Path | None,
    preprocessor_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    """Build the ``dataset_stats`` dict that :func:`make_vla_jepa_preprocessors` expects.

    Same shape as SmolVLA's loader, but every stat field is carried over rather than mean/std
    alone: VLA-JEPA normalizes actions with MIN_MAX, so ``min``/``max`` must survive.

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
