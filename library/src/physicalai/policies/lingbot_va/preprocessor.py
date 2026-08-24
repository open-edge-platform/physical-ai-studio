# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre/post-processing for the LingBot-VA policy.

The model consumes raw camera images (it resizes and VAE-encodes them itself), so the
preprocessor only has to do two things: line the configured camera keys up with whatever
the batch actually carries, and put actions in the model's normalized space.

Action normalization is **symmetric**: the preprocessor maps ground-truth actions into the
model's ``[-1, 1]`` space with the checkpoint's per-channel q01/q99, and the postprocessor
maps predictions back to physical units with the same statistics. (The upstream LeRobot
integration normalizes only on the way out; normalizing on the way in as well is what keeps
the training target in the same space the model predicts in.)
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, cast

import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.observation import ACTION, IMAGES
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from collections.abc import Sequence

_CAMERA_PREFIXES = ("observation.images.", "observation.image.", "observation.", "images.")


def camera_basename(key: str) -> str:
    """Strip the dataset-specific prefix from a camera key.

    Args:
        key: A camera key in either LeRobot (``"observation.images.image"``) or Studio
            (``"images.image"``, ``"image"``) form.

    Returns:
        The bare camera name.
    """
    for prefix in _CAMERA_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def resolve_camera_keys(batch: dict[str, Any], obs_cam_keys: Sequence[str]) -> list[str]:
    """Map configured camera keys onto the keys the batch actually carries.

    LingBot-VA checkpoints name their cameras in LeRobot form
    (``"observation.images.image"``) while Studio batches flatten them to
    ``"images.image"``. Both spellings — and the bare camera name — resolve here, so a
    checkpoint config works unchanged against a Studio datamodule or gym.

    Args:
        batch: Flattened batch dict.
        obs_cam_keys: The configured camera keys, in order.

    Returns:
        The batch keys corresponding to ``obs_cam_keys``, in the same order.

    Raises:
        KeyError: If a configured camera is not present in the batch.
    """
    resolved: list[str] = []
    for key in obs_cam_keys:
        base = camera_basename(key)
        candidates = (key, f"{IMAGES}.{base}", base, f"observation.images.{base}")
        match = next((candidate for candidate in candidates if candidate in batch), None)
        if match is None:
            available = [k for k in batch if k.startswith(f"{IMAGES}.") or "image" in k.lower()]
            msg = f"Camera {key!r} not found in batch. Tried {list(candidates)}; available image keys: {available}"
            raise KeyError(msg)
        resolved.append(match)
    return resolved


def _norm_map_for_mode(mode: str) -> dict[FeatureType, NormalizationType]:
    """Return the action normalization mapping for the given mode string.

    Args:
        mode: ``"QUANTILES"`` or ``"MEAN_STD"``.

    Returns:
        Mapping from ``FeatureType`` to ``NormalizationType``.
    """
    return {FeatureType.ACTION: NormalizationType(mode)}


class LingBotVAPreprocessor(torch.nn.Module):
    """Normalize ground-truth actions into the model's action space.

    Camera images are passed through untouched: the model resizes them to
    ``(config.height, config.width)`` and scales them to ``[-1, 1]`` right before the VAE.

    Args:
        features: Feature definitions carrying the normalization statistics.
        normalization_mode: ``"QUANTILES"`` (default) or ``"MEAN_STD"``.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        normalization_mode: str = "QUANTILES",
    ) -> None:
        """Build the action normalizer."""
        super().__init__()
        self.normalization_mode = normalization_mode
        norm_map = _norm_map_for_mode(normalization_mode)
        if features:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_normalizer: torch.nn.Module = FeatureNormalizeTransform(action_features, norm_map)
        else:
            self._action_normalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize the batch's actions, if present.

        Args:
            batch: Flattened batch dict.

        Returns:
            A shallow copy of the batch with normalized actions.
        """
        batch = copy(batch)
        if batch.get(ACTION) is not None:
            batch[ACTION] = self._action_normalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


class LingBotVAPostprocessor(torch.nn.Module):
    """Map predicted actions from the model's ``[-1, 1]`` space back to physical units.

    Args:
        features: Feature definitions carrying the normalization statistics.
        normalization_mode: ``"QUANTILES"`` (default) or ``"MEAN_STD"``.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        normalization_mode: str = "QUANTILES",
    ) -> None:
        """Build the action denormalizer."""
        super().__init__()
        norm_map = _norm_map_for_mode(normalization_mode)
        if features:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer: torch.nn.Module = FeatureNormalizeTransform(
                action_features,
                norm_map,
                inverse=True,
            )
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Denormalize the batch's actions.

        Args:
            batch: Dict holding an ``action`` entry.

        Returns:
            A shallow copy of the batch with denormalized actions.
        """
        batch = dict(batch)
        if batch.get(ACTION) is not None:
            batch[ACTION] = self._action_denormalizer({ACTION: batch[ACTION]})[ACTION]
        return batch


def build_action_features(
    stats: dict[str, dict[str, Any]] | None,
    used_action_channel_ids: Sequence[int] | None = None,
) -> dict[str, Feature]:
    """Build the action :class:`~physicalai.data.Feature` from dataset statistics.

    Statistics wider than the policy's action space (for example a 30-channel dataset
    against a 7-channel LIBERO checkpoint) are sliced down to
    ``used_action_channel_ids`` so the buffers match the predicted action width.

    Args:
        stats: Dataset statistics keyed by feature name.
        used_action_channel_ids: Action channels this checkpoint drives.

    Returns:
        Dict with an ``action`` feature, or empty if no action statistics were found.
    """
    if not stats:
        return {}

    entry = next(
        (stat for name, stat in stats.items() if name == ACTION or name.rsplit(".", 1)[-1] == ACTION),
        None,
    )
    if entry is None:
        return {}

    def _select(values: Any) -> list[float] | None:  # noqa: ANN401
        if values is None:
            return None
        values = list(values)
        if used_action_channel_ids is not None and len(values) > len(used_action_channel_ids):
            values = [values[i] for i in used_action_channel_ids]
        return [float(v) for v in values]

    norm_data = NormalizationParameters(
        mean=_select(entry.get("mean")),
        std=_select(entry.get("std")),
        q01=_select(entry.get("q01")),
        q99=_select(entry.get("q99")),
    )
    shape = _select(entry.get("q01")) or _select(entry.get("mean"))
    feature_shape = (len(shape),) if shape else cast("tuple[int, ...]", tuple(entry.get("shape", ())))

    return {
        ACTION: Feature(
            name=ACTION,
            ftype=FeatureType.ACTION,
            shape=feature_shape,
            normalization_data=norm_data,
        ),
    }


def make_lingbot_va_preprocessors(
    stats: dict[str, dict[str, Any]] | None = None,
    *,
    used_action_channel_ids: Sequence[int] | None = None,
    normalization_mode: str = "QUANTILES",
) -> tuple[LingBotVAPreprocessor, LingBotVAPostprocessor]:
    """Create the preprocessor / postprocessor pair for LingBot-VA.

    Args:
        stats: Dataset statistics (from a checkpoint or a training dataset).
        used_action_channel_ids: Action channels this checkpoint drives.
        normalization_mode: ``"QUANTILES"`` (default) or ``"MEAN_STD"``.

    Returns:
        Tuple of ``(preprocessor, postprocessor)``.
    """
    features = build_action_features(stats, used_action_channel_ids)
    return (
        LingBotVAPreprocessor(features, normalization_mode=normalization_mode),
        LingBotVAPostprocessor(features, normalization_mode=normalization_mode),
    )


__all__ = [
    "LingBotVAPostprocessor",
    "LingBotVAPreprocessor",
    "build_action_features",
    "camera_basename",
    "make_lingbot_va_preprocessors",
    "resolve_camera_keys",
]
