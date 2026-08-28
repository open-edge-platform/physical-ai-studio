# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""PAS-native preprocessing building blocks for the RLDX-1 policy.

This module hosts the incrementally-migrated, PAS-style replacements for the
vendored upstream ``RLDXProcessor`` / ``RLDXDataCollator`` pipeline (see
``components/processing/processing_rldx.py``). The blocks are plain functions
that the policy preprocessor composes in ``forward`` -- mirroring the
conventions used by the other first-party policies (``pi05``, ``smolvla``):
batched torch, library normalization API, export-safe buffers, no extra
``nn.Module`` ceremony.

Stage 1 scope: state/action normalization. :func:`build_state_action_features`
and :func:`build_state_action_norm_map` feed the shared
:class:`~physicalai.policies.utils.normalization.FeatureNormalizeTransform`;
:func:`clip_state_action` applies the vendored ``clip_outliers`` step. Together
they reproduce the vendored :class:`StateActionProcessor` single-group
normalization (per-feature min/max or q01/q99 mapped to ``[-1, 1]`` with
clipping).

Stage 2 scope: state/action padding + action mask. :func:`pad_state_action`
reproduces the in-``__call__`` padding from the vendored ``RLDXProcessor``
(zero-pad state to ``max_state_dim``; zero-pad the action chunk to
``max_action_horizon`` x ``max_action_dim`` and build the validity mask).

Stage 3 scope: image geometry. The deterministic area-budget resize + ``m``-
aligned center crop lives in :class:`~physicalai.policies.rldx1.augmentations.AspectAreaResizeAndCrop`
(torchvision-based, same transform at train and eval time; no train-time
stochastic augmentation).

Stage 4 scope: Qwen VLM tokenization orchestration. :func:`formalize_language`
ports the vendored lowercase + punctuation strip; :func:`build_qwen_conversation`
builds the text/image chat turn; :func:`tokenize_vlm_batch` runs the chat
template, Qwen vision tiler (``process_vision_info``) and HF processor over a
batch -- replacing ``RLDXProcessor._apply_vlm_processing`` /
``RLDXDataCollator._collate_vlm_content``. The HF Qwen processor itself is
retained (loaded lazily, pinned revision, ``trust_remote_code=False`` by the
owning preprocessor).

Out of scope (deferred): relative-action pose math, sin/cos state encoding,
multi-joint-group modality configs, physics streams. Those still route through
the vendored processor until later migration stages.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.observation import ACTION, STATE
from physicalai.policies.utils.normalization import NormalizationType

from .components.processing.qwen_vision_process import process_vision_info

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.feature_extraction_utils import BatchFeature

# Output key for the action validity mask (matches the vendored collator key).
ACTION_MASK = "action_mask"

# Native Observation field names the normalizer operates on, plus the LeRobot
# aliases that may arrive in a flat batch dict.
_STATE_KEYS = (STATE, "observation.state")
_ACTION_KEYS = (ACTION, "action")

# Stat sub-keys expected from Studio dataset statistics.
_STAT_MIN = "min"
_STAT_MAX = "max"
_STAT_MEAN = "mean"
_STAT_STD = "std"
_STAT_Q01 = "q01"
_STAT_Q99 = "q99"


def _coerce_vector(raw: Any, key: str) -> list[float] | None:  # noqa: ANN401
    """Return ``raw[key]`` flattened to a ``list[float]`` or ``None``.

    Args:
        raw: A stat mapping (e.g. ``{"min": [...], "max": [...]}``).
        key: Stat sub-key to extract.

    Returns:
        The flattened vector, or ``None`` when the sub-key is absent.
    """
    if not isinstance(raw, dict):
        return None
    value = raw.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).ravel().tolist()


def _build_feature(stat: dict[str, Any] | None, ftype: FeatureType, name: str) -> Feature | None:
    """Build a :class:`Feature` carrying min/max/q01/q99 normalization stats.

    Args:
        stat: Raw stat mapping for this modality, or ``None``.
        ftype: Feature type (``STATE`` or ``ACTION``).
        name: Feature name used to match batch keys.

    Returns:
        A populated :class:`Feature`, or ``None`` when ``stat`` carries no
        numeric vector.
    """
    if stat is None:
        return None

    mn = _coerce_vector(stat, _STAT_MIN)
    mx = _coerce_vector(stat, _STAT_MAX)
    q01 = _coerce_vector(stat, _STAT_Q01)
    q99 = _coerce_vector(stat, _STAT_Q99)
    mean = _coerce_vector(stat, _STAT_MEAN)
    std = _coerce_vector(stat, _STAT_STD)

    dim_src = next((v for v in (mn, mx, q01, q99, mean, std) if v is not None), None)
    if dim_src is None:
        return None

    norm_data = NormalizationParameters(mean=mean, std=std, min=mn, max=mx, q01=q01, q99=q99)
    return Feature(name=name, ftype=ftype, shape=(len(dim_src),), normalization_data=norm_data)


def _first_stat(stats: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """Return the first present stat entry among ``keys``.

    Args:
        stats: Studio dataset statistics keyed by feature name.
        keys: Candidate feature names, in priority order.

    Returns:
        The matching stat mapping, or ``None``.
    """
    if not stats:
        return None
    for key in keys:
        entry = stats.get(key)
        if entry is not None:
            return entry
    return None


def build_state_action_features(
    stats: dict[str, dict[str, list[float]]] | None,
) -> dict[str, Feature]:
    """Build the state/action :class:`Feature` map from Studio dataset stats.

    Args:
        stats: Dataset statistics keyed by feature name (``state`` /
            ``observation.state`` and ``action``).

    Returns:
        A mapping ``{name: Feature}`` for the present state/action modalities.
        Empty when no usable stats are found.
    """
    features: dict[str, Feature] = {}

    state_feature = _build_feature(_first_stat(stats, _STATE_KEYS), FeatureType.STATE, STATE)
    if state_feature is not None:
        features[STATE] = state_feature

    action_feature = _build_feature(_first_stat(stats, _ACTION_KEYS), FeatureType.ACTION, ACTION)
    if action_feature is not None:
        features[ACTION] = action_feature

    return features


def build_state_action_norm_map(*, use_percentiles: bool) -> dict[FeatureType, NormalizationType]:
    """Return the state/action normalization-type map.

    Args:
        use_percentiles: Use q01/q99 bounds (``QUANTILES``) instead of min/max
            (``MIN_MAX``). Mirrors the vendored ``use_percentiles`` flag.

    Returns:
        Mapping from ``FeatureType`` to ``NormalizationType`` for state/action.
    """
    norm_type = NormalizationType.QUANTILES if use_percentiles else NormalizationType.MIN_MAX
    return {
        FeatureType.STATE: norm_type,
        FeatureType.ACTION: norm_type,
    }


def clip_state_action(batch: dict[str, Any]) -> dict[str, Any]:
    """Clip normalized ``state`` / ``action`` to ``[-1, 1]`` in-place.

    Mirrors the vendored ``clip_outliers`` step that follows
    ``normalize_values_minmax``. Apply after a :class:`FeatureNormalizeTransform`
    forward pass.

    Args:
        batch: Batch dict with normalized state/action tensors.

    Returns:
        The same batch dict with state/action clipped.

    Note:
        Degenerate dimensions (``max == min``, a constant channel) need
        :func:`zero_degenerate_state_action` to reach parity: the vendored
        pipeline masks them out and emits ``0`` while
        :class:`FeatureNormalizeTransform` emits ``2*(x-lo)/1e-8 - 1 == -1``
        (then clipped to ``-1``). Non-degenerate dimensions agree exactly.
    """
    for key in (STATE, ACTION):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            batch[key] = value.clamp(-1.0, 1.0)
    return batch


def build_state_action_degenerate_masks(
    features: dict[str, Feature],
    *,
    use_percentiles: bool,
) -> dict[str, torch.Tensor]:
    """Build boolean masks marking degenerate (constant) state/action channels.

    A channel is degenerate when its normalization bounds collapse
    (``q99 == q01`` for percentiles, else ``max == min``). The vendored
    ``normalize_values_minmax`` masks these with ``~np.isclose(max, min)`` and
    leaves them at ``0``; PAS's :class:`FeatureNormalizeTransform` instead maps
    them to ``-1``. Feed the returned masks to
    :func:`zero_degenerate_state_action` to restore upstream parity.

    Args:
        features: State/action feature map (see
            :func:`build_state_action_features`).
        use_percentiles: Match the paired normalizer -- compare ``q01``/``q99``
            when ``True``, else ``min``/``max``.

    Returns:
        ``{name: bool mask}`` of shape ``(feature_dim,)``; ``True`` marks a
        degenerate channel. Features whose bounds are missing are omitted.
    """
    masks: dict[str, torch.Tensor] = {}
    for name, feature in features.items():
        norm = feature.normalization_data
        if norm is None:
            continue
        lo, hi = (norm.q01, norm.q99) if use_percentiles else (norm.min, norm.max)
        if lo is None or hi is None:
            continue
        lo_arr = np.asarray(lo, dtype=np.float64)
        hi_arr = np.asarray(hi, dtype=np.float64)
        masks[name] = torch.from_numpy(np.isclose(hi_arr, lo_arr))
    return masks


def zero_degenerate_state_action(
    batch: dict[str, Any],
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Zero degenerate (constant) state/action channels in-place.

    Restores parity with the vendored ``normalize_values_minmax`` mask path,
    which leaves constant channels at ``0`` (see
    :func:`build_state_action_degenerate_masks`). Apply after
    :func:`clip_state_action`.

    Args:
        batch: Batch dict with normalized state/action tensors.
        masks: Degenerate-channel masks keyed by ``state`` / ``action``.

    Returns:
        The same batch dict with degenerate channels set to ``0``.
    """
    for key in (STATE, ACTION):
        mask = masks.get(key)
        value = batch.get(key)
        if mask is not None and isinstance(value, torch.Tensor):
            value[..., mask.to(value.device)] = 0.0
    return batch


def _pad_last_dim(tensor: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Zero-pad the last dimension of ``tensor`` up to ``new_dim``.

    Args:
        tensor: Input tensor.
        new_dim: Target size of the last dimension.

    Returns:
        The padded tensor (unchanged when already at least ``new_dim`` wide).
    """
    if tensor.shape[-1] >= new_dim:
        return tensor
    return F.pad(tensor, (0, new_dim - tensor.shape[-1]))


def pad_state_action(
    batch: dict[str, Any],
    *,
    max_state_dim: int,
    max_action_dim: int,
    max_action_horizon: int,
) -> dict[str, Any]:
    """Pad normalized state/action to fixed dims and build the action mask.

    Reproduces the padding block of the vendored ``RLDXProcessor.__call__``
    (``components/processing/processing_rldx.py``) as batched torch. RLDX's
    MSAT action head requires the explicit state time axis, the horizon /
    feature padding, and the validity mask; other policies only pad the action
    feature dim and need none of this.

    - ``state``: ``(B, state_dim)`` or ``(B, 1, state_dim)`` -> zero-padded to
      ``(B, 1, max_state_dim)``.
    - ``action``: ``(B, T, action_dim)`` -> zero-padded to
      ``(B, max_action_horizon, max_action_dim)``.
    - ``action_mask``: ``(B, max_action_horizon, max_action_dim)``, ``1`` over
      the valid ``[0:T)`` rows and ``[0:action_dim)`` columns, ``0`` elsewhere.

    The unpadded ``T`` and ``action_dim`` are read from the input action tensor,
    matching the vendored logic exactly. At inference (no ``action`` in the
    batch) only the state is padded.

    Args:
        batch: Batch dict with normalized ``state`` and optionally ``action``.
        max_state_dim: Padded state feature dimension.
        max_action_dim: Padded action feature dimension.
        max_action_horizon: Padded number of action steps.

    Returns:
        The same batch dict with padded ``state``, ``action`` and the new
        ``action_mask`` entry (the latter two only when ``action`` is present).
    """
    state = batch.get(STATE)
    if isinstance(state, torch.Tensor):
        if state.ndim == 2:  # noqa: PLR2004  (B, state_dim) -> (B, 1, state_dim)
            state = state.unsqueeze(1)
        batch[STATE] = _pad_last_dim(state, max_state_dim)

    action = batch.get(ACTION)
    if isinstance(action, torch.Tensor):
        if action.ndim == 2:  # noqa: PLR2004  (B, action_dim) -> (B, 1, action_dim)
            action = action.unsqueeze(1)
        horizon = action.shape[-2]
        action_dim = action.shape[-1]

        padded = _pad_last_dim(action, max_action_dim)
        if horizon < max_action_horizon:
            # Pad the time axis (dim -2): (left_d, right_d, left_t, right_t).
            padded = F.pad(padded, (0, 0, 0, max_action_horizon - horizon))

        mask = torch.ones(
            padded.shape[0],
            max_action_horizon,
            max_action_dim,
            dtype=padded.dtype,
            device=padded.device,
        )
        mask[:, horizon:, :] = 0
        mask[:, :, action_dim:] = 0

        batch[ACTION] = padded
        batch[ACTION_MASK] = mask

    return batch


# Qwen vision tiler patch size used by the vendored collator (image_patch_size).
_VLM_IMAGE_PATCH_SIZE = 16


def formalize_language(text: str) -> str:
    r"""Lowercase and strip punctuation from an instruction.

    Exact port of the vendored ``formalize_language`` step
    (``re.sub(r"[^\w\s]", "", text.lower())``).

    Args:
        text: Raw task instruction.

    Returns:
        The lowercased, punctuation-stripped instruction.
    """
    return re.sub(r"[^\w\s]", "", text.lower())


def build_qwen_conversation(
    images: list[Any],
    text: str,
    *,
    image_first: bool = False,
) -> list[dict[str, Any]]:
    """Build a single-turn Qwen chat conversation from images + text.

    Mirrors the conversation assembled in
    ``RLDXProcessor._apply_vlm_processing``. With ``image_first=False`` (the
    RLDX default) the text block precedes the image blocks.

    Args:
        images: ``PIL.Image`` objects (already resized by Stage 3 geometry).
        text: Instruction text (already passed through
            :func:`formalize_language` when enabled).
        image_first: Place image blocks before the text block.

    Returns:
        A one-element conversation list with a single ``user`` turn.
    """
    image_blocks: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    text_block = {"type": "text", "text": text}
    content = [*image_blocks, text_block] if image_first else [text_block, *image_blocks]
    return [{"role": "user", "content": content}]


def tokenize_vlm_batch(
    processor: ProcessorMixin,
    conversations: list[list[dict[str, Any]]],
    *,
    image_patch_size: int = _VLM_IMAGE_PATCH_SIZE,
) -> BatchFeature:
    """Tokenize a batch of Qwen conversations into VLM model inputs.

    Replaces ``RLDXDataCollator._collate_vlm_content``: applies the chat
    template per sample, runs the Qwen vision tiler
    (``process_vision_info``) over the batch, then calls the HF processor with
    ``do_resize=False`` (the tiler already produced patch-aligned images) and
    left padding.

    Args:
        processor: A loaded HF Qwen processor. The caller is responsible for
            pinning the revision, disabling ``trust_remote_code``, and setting
            ``tokenizer.padding_side = "left"``.
        conversations: One conversation per batch sample (see
            :func:`build_qwen_conversation`).
        image_patch_size: Patch size for the Qwen vision tiler.

    Returns:
        The processor output (``input_ids``, ``attention_mask``,
        ``pixel_values``, ``image_grid_thw``, ...).
    """
    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
    image_inputs = process_vision_info(conversations, image_patch_size=image_patch_size)

    processor_kwargs: dict[str, Any] = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "do_resize": False,
    }
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    return processor(**processor_kwargs)
