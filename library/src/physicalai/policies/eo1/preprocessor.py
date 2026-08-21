# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

"""Pre- and postprocessing for EO-1.

Studio equivalent of the normalization half of LeRobot's ``lerobot.policies.eo1.processor_eo1``
pipeline. The conversation-template and Qwen-tokenizer steps live next to the model instead, in
:mod:`physicalai.policies.eo1.components.qwen_interface`, because their output has to land on the
backbone's device.

What is left here is normalization:

    raw -> normalize -> model -> denormalize

Visual observations are passed through untouched; the Qwen interface quantizes them to uint8 on
its way into the image processor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.data.observation import ACTION, STATE
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import EO1Config

_REQUIRED_STAT_FIELDS = {
    NormalizationType.MEAN_STD: ("mean", "std"),
    NormalizationType.MIN_MAX: ("min", "max"),
    NormalizationType.QUANTILES: ("q01", "q99"),
}


def _norm_type(name: str) -> NormalizationType:
    """Resolve a normalization type from its config string.

    Args:
        name: Normalization name, e.g. ``"MEAN_STD"``.

    Returns:
        The matching :class:`NormalizationType`.

    Raises:
        ValueError: If the name is not a known normalization type.
    """
    try:
        return NormalizationType(name.upper())
    except ValueError as e:
        supported = ", ".join(t.value for t in NormalizationType)
        msg = f"Unknown normalization '{name}'. Supported: {supported}."
        raise ValueError(msg) from e


def features_from_stats(stats: dict[str, dict[str, Any]] | None) -> dict[str, Feature]:
    """Build state/action feature descriptors from dataset statistics.

    Args:
        stats: Dataset statistics keyed by feature name (e.g. ``observation.state``, ``action``).

    Returns:
        Mapping from the raw feature name to its :class:`~physicalai.data.Feature`.
    """
    features: dict[str, Feature] = {}
    for key, stat in (stats or {}).items():
        if ACTION in key:
            feature_type = FeatureType.ACTION
        elif STATE in key:
            feature_type = FeatureType.STATE
        else:
            continue
        features[str(stat["name"])] = Feature(
            name=str(stat["name"]),
            ftype=feature_type,
            shape=cast("tuple[int, ...]", stat["shape"]),
            normalization_data=NormalizationParameters(
                mean=cast("list[float]", stat.get("mean")),
                std=cast("list[float]", stat.get("std")),
                min=cast("list[float]", stat.get("min")),
                max=cast("list[float]", stat.get("max")),
                q01=cast("list[float]", stat.get("q01")),
                q99=cast("list[float]", stat.get("q99")),
            ),
        )
    return features


def _validate_stat_fields(features: dict[str, Feature], ftype: FeatureType, norm: NormalizationType) -> None:
    """Fail early when the statistics lack the fields the normalization needs.

    Args:
        features: Feature descriptors built from the dataset statistics.
        ftype: The feature type to check.
        norm: The normalization configured for that feature type.

    Raises:
        ValueError: If a feature of this type is missing a required statistic.
    """
    required = _REQUIRED_STAT_FIELDS.get(norm)
    if required is None:
        return
    for name, feature in features.items():
        if feature.ftype is not ftype or feature.normalization_data is None:
            continue
        missing = [field for field in required if getattr(feature.normalization_data, field) is None]
        if missing:
            msg = (
                f"Feature '{name}' has no {'/'.join(missing)} statistics, which {norm.value} "
                f"normalization requires. Provide dataset stats carrying them, or configure a "
                f"different normalization for this feature type."
            )
            raise ValueError(msg)


class EO1Preprocessor(torch.nn.Module):
    """Preprocessor for EO-1 model inputs.

    Args:
        features: Feature descriptors used to build the normalization buffers. When None, no
            normalization is applied.
        state_normalization: Normalization applied to the robot state.
        action_normalization: Normalization applied to actions.

    Example:
        >>> preprocessor = EO1Preprocessor(features=features)
        >>> batch = preprocessor(raw_batch)
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        *,
        state_normalization: str = "MEAN_STD",
        action_normalization: str = "MEAN_STD",
    ) -> None:
        """Initialize the preprocessor.

        Args:
            features: Feature descriptors used to build the normalization buffers.
            state_normalization: Normalization applied to the robot state.
            action_normalization: Normalization applied to actions.
        """
        super().__init__()

        if features is not None:
            norm_map = {
                FeatureType.STATE: _norm_type(state_normalization),
                FeatureType.ACTION: _norm_type(action_normalization),
            }
            self._state_action_normalizer: torch.nn.Module = FeatureNormalizeTransform(features, norm_map)
        else:
            self._state_action_normalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize the robot state and the ground-truth actions.

        Args:
            batch: Flattened observation dict with ``images.*``, ``state``, ``action`` and ``task``.

        Returns:
            The processed batch, ready for :class:`~physicalai.policies.eo1.EO1Model`.
        """
        return self._state_action_normalizer(dict(batch))


class EO1Postprocessor(torch.nn.Module):
    """Postprocessor for EO-1 model outputs.

    Args:
        features: Feature descriptors used to build the denormalization buffers. When None, no
            denormalization is applied.
        action_normalization: Normalization that was applied to actions.
    """

    def __init__(
        self,
        features: dict[str, Feature] | None = None,
        *,
        action_normalization: str = "MEAN_STD",
    ) -> None:
        """Initialize the postprocessor.

        Args:
            features: Feature descriptors used to build the denormalization buffers.
            action_normalization: Normalization that was applied to actions.
        """
        super().__init__()

        if features is not None:
            action_features = {k: v for k, v in features.items() if v.ftype == FeatureType.ACTION}
            self._action_denormalizer: torch.nn.Module = FeatureNormalizeTransform(
                action_features,
                {FeatureType.ACTION: _norm_type(action_normalization)},
                inverse=True,
            )
        else:
            self._action_denormalizer = torch.nn.Identity()

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Denormalize actions back into the dataset's action space.

        Args:
            batch: Dict that may contain an ``action`` tensor produced by the model.

        Returns:
            A dict with the action converted back to the dataset's action space.
        """
        batch = dict(batch)
        if batch.get(ACTION) is None:
            return batch
        return self._action_denormalizer(batch)


def make_eo1_preprocessors(
    config: EO1Config,
    stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[EO1Preprocessor, EO1Postprocessor]:
    """Create a matched preprocessor / postprocessor pair.

    Args:
        config: Policy configuration.
        stats: Dataset statistics used to build the normalization buffers.

    Returns:
        Tuple of (preprocessor, postprocessor).
    """
    features = features_from_stats(stats)
    if features:
        _validate_stat_fields(features, FeatureType.STATE, _norm_type(config.state_normalization))
        _validate_stat_fields(features, FeatureType.ACTION, _norm_type(config.action_normalization))

    preprocessor = EO1Preprocessor(
        features=features or None,
        state_normalization=config.state_normalization,
        action_normalization=config.action_normalization,
    )
    postprocessor = EO1Postprocessor(
        features=features or None,
        action_normalization=config.action_normalization,
    )
    return preprocessor, postprocessor
