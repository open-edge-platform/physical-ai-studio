# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Input and output processing for Patch Policy."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from itertools import starmap
from math import prod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as tvf

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import PatchPolicyConfig

_BATCHED_IMAGE_NDIM = 4
_TEMPORAL_IMAGE_NDIM = 5
_ACTION_NDIM = 3


def resolve_camera_name(feature: Feature) -> str:
    """Return the canonical camera name for a visual feature.

    Returns:
        Camera name without observation or image prefixes.

    Raises:
        ValueError: If the feature has no usable name.
    """
    name = (feature.name or "").removeprefix(f"observation.{IMAGES}.").removeprefix(f"{IMAGES}.")
    if not name:
        msg = "Patch Policy visual features must have non-empty camera names."
        raise ValueError(msg)
    return name


def _normalizer(
    feature: Feature,
    key: str,
    *,
    mode: NormalizationType = NormalizationType.MEAN_STD,
    inverse: bool = False,
) -> FeatureNormalizeTransform:
    """Build a normalizer using complete feature statistics when available.

    Returns:
        Feature transform using the requested mode or identity when statistics are missing.
    """
    stats = feature.normalization_data
    has_mean_std = stats is not None and stats.mean is not None and stats.std is not None
    stats_available = (
        has_mean_std
        if mode is NormalizationType.MEAN_STD
        else stats is not None and stats.min is not None and stats.max is not None
    )
    mode = (
        mode
        if stats_available and feature.ftype in {FeatureType.VISUAL, FeatureType.ACTION}
        else NormalizationType.IDENTITY
    )
    feature_type = feature.ftype
    features = {key: feature} if mode is not NormalizationType.IDENTITY else {}
    norm_map = {feature_type: mode} if feature_type is not None else {}
    return FeatureNormalizeTransform(features, norm_map, inverse=inverse)


def _warn_missing_action_normalization(features: list[Feature], operation: str) -> None:
    missing_features = []
    for index, feature in enumerate(features):
        stats = feature.normalization_data
        if stats is None or stats.mean is None or stats.std is None:
            missing_features.append(feature.name or f"index {index}")
    if missing_features:
        warnings.warn(
            f"Missing action mean/std normalization data for {operation} on: {', '.join(missing_features)}. "
            "Identity normalization will be used.",
            stacklevel=3,
        )


def _normalize_features(
    values: Tensor,
    features: list[Feature],
    normalizers: nn.ModuleList,
) -> Tensor:
    """Apply shared feature normalizers to slices of a concatenated tensor.

    Returns:
        Concatenated normalized feature values.

    Raises:
        ValueError: If a feature has no shape or the configured total width differs from the tensor.
    """
    normalized = values.clone()
    offset = 0
    for index, (feature, normalizer) in enumerate(zip(features, normalizers, strict=True)):
        if not feature.shape:
            msg = f"Feature {feature.name!r} has no shape."
            raise ValueError(msg)
        end = offset + prod(feature.shape)
        key = f"feature_{index}"
        piece = {key: normalized[..., offset:end]}
        normalizer(piece)
        normalized[..., offset:end] = piece[key]
        offset = end
    if values.shape[-1] != offset:
        msg = f"Feature width {values.shape[-1]} does not match configured width {offset}."
        raise ValueError(msg)
    return normalized


def _gather_flattened_or_unflattened_key(batch: dict[str, Any], field: str) -> Mapping[str, Any]:
    """Gather a nested or flattened observation field into a keyed mapping.

    Returns:
        Field values keyed by their names without the field prefix.
    """
    if field in batch and isinstance(batch[field], Mapping):
        return batch[field]
    prefixed_keys = Observation.get_flattened_keys(batch, field)
    prefix = f"{field}."
    return {key.removeprefix(prefix): batch[key] for key in prefixed_keys}


class PatchPolicyPreprocessor(nn.Module):
    """Order camera views, resize and normalize images, and normalize action targets.

    Args:
        input_features: Observation features used to determine camera order.
        output_features: Action features and their normalization statistics.
        image_size: Target image height and width.
        n_obs_steps: Required number of temporal observation frames.
    """

    def __init__(
        self,
        input_features: list[Feature],
        output_features: list[Feature],
        image_size: tuple[int, int],
        n_obs_steps: int = 1,
        action_normalization: NormalizationType = NormalizationType.MEAN_STD,
    ) -> None:
        """Initialize camera and action transforms.

        Raises:
            ValueError: If the observation window is not positive or camera names are duplicated.
        """
        super().__init__()
        if n_obs_steps <= 0:
            msg = "n_obs_steps must be positive."
            raise ValueError(msg)
        self.visual_features = [feature for feature in input_features if feature.ftype is FeatureType.VISUAL]
        self.output_features = output_features
        if action_normalization is NormalizationType.MEAN_STD:
            _warn_missing_action_normalization(output_features, "normalization")
        self.image_size = image_size
        self.n_obs_steps = n_obs_steps
        self.action_normalization = action_normalization
        camera_names = [resolve_camera_name(feature) for feature in self.visual_features]
        if len(camera_names) != len(set(camera_names)):
            msg = f"Patch Policy camera names must be unique, got {camera_names!r}."
            raise ValueError(msg)
        self._image_normalizers = nn.ModuleList(
            starmap(_normalizer, zip(self.visual_features, (f"{IMAGES}.{name}" for name in camera_names), strict=True)),
        )
        self._action_normalizers = nn.ModuleList(
            _normalizer(feature, f"feature_{index}", mode=action_normalization)
            for index, feature in enumerate(output_features)
        )

    @staticmethod
    def _image(images: Mapping[str, Any] | Tensor, feature: Feature) -> Tensor:
        """Resolve one configured camera image from an observation.

        Args:
            images: A shared image tensor or mapping of camera names to tensors.
            feature: Visual feature identifying the requested camera.

        Returns:
            The selected image tensor.

        Raises:
            KeyError: If the configured camera is absent from an image mapping.
            ValueError: If a shared image tensor has too few dimensions.
        """
        if isinstance(images, Tensor):
            if len(images.shape) < _BATCHED_IMAGE_NDIM:
                msg = "Visual input must include batch, channel, height, and width dimensions."
                raise ValueError(msg)
            return images
        canonical_name = resolve_camera_name(feature)
        names = [feature.name or "", canonical_name, f"{IMAGES}.{canonical_name}"]
        for name in names:
            if name in images and isinstance(images[name], Tensor):
                return images[name]
        msg = f"Missing visual feature {feature.name!r} in Observation.images."
        raise KeyError(msg)

    @staticmethod
    def _with_time_axis(image: Tensor, feature: Feature, n_obs_steps: int) -> Tensor:
        if image.ndim == _BATCHED_IMAGE_NDIM and n_obs_steps == 1:
            return image[:, None]
        if image.ndim == _TEMPORAL_IMAGE_NDIM and image.shape[1] == n_obs_steps:
            return image
        if image.ndim == _TEMPORAL_IMAGE_NDIM:
            msg = f"Expected {n_obs_steps} observation timestep(s), got {image.shape[1]} for {feature.name!r}."
            raise ValueError(msg)
        msg = (
            f"Expected image shape [B, {n_obs_steps}, C, H, W] for key {feature.name!r}, "
            f"got {tuple(image.shape)}. [B, C, H, W] is only accepted when n_obs_steps=1."
        )
        raise ValueError(msg)

    @staticmethod
    def _actions(batch: dict[str, Any]) -> Tensor | None:
        if ACTION in batch:
            actions = batch[ACTION]
        elif Observation.get_flattened_keys(batch, ACTION):
            msg = "Expected actions as a tensor, not flattened action fields."
            raise TypeError(msg)
        else:
            return None
        if actions is None:
            return None
        if not isinstance(actions, Tensor):
            msg = f"Expected actions to be a tensor or None, got {type(actions)}."
            raise TypeError(msg)
        if actions.ndim != _ACTION_NDIM:
            msg = f"Expected action shape [B, CHUNK, DIM], got {tuple(actions.shape)}."
            raise ValueError(msg)
        return actions

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Normalize each action feature with its configured statistics.

        Args:
            actions: Concatenated action tensor.

        Returns:
            The normalized action tensor.

        """
        return _normalize_features(actions, self.output_features, self._action_normalizers)

    def forward(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Convert an observation mapping into model inputs.

        Args:
            batch: Observation values containing images and optional actions.

        Returns:
            Model inputs containing per-camera images and optional action metadata.

        Raises:
            TypeError: If images or required actions are not tensors in a supported layout.
        """
        # Gather camera fields from either flattened or nested observations.
        batch = dict(batch)
        images = _gather_flattened_or_unflattened_key(batch, IMAGES)
        if not images:
            msg = "Observation must contain at least one image tensor."
            raise TypeError(msg)
        if len(images) == 1 and IMAGES in images:
            images = images[IMAGES]
        if images is None:
            msg = "Observation.images cannot be None."
            raise TypeError(msg)

        processed: dict[str, Any] = {}
        for feature, normalizer in zip(self.visual_features, self._image_normalizers, strict=True):
            camera_name = resolve_camera_name(feature)
            image = self._with_time_axis(self._image(images, feature), feature, self.n_obs_steps)
            image = tvf.to_dtype(image, torch.float32, scale=True)
            processed[f"{IMAGES}.{camera_name}"] = tvf.resize(
                image,
                list(self.image_size),
                antialias=False,
            )
            normalizer(processed)

        # Keep actions as [B, CHUNK, DIM] while normalizing configured features.
        actions = self._actions(batch)
        if isinstance(actions, Tensor):
            processed["action"] = self.normalize_actions(actions)
        action_is_pad = batch.get("extra.action_is_pad")
        extra = batch.get("extra")
        if action_is_pad is None and isinstance(extra, Mapping):
            action_is_pad = extra.get("action_is_pad")
        if isinstance(action_is_pad, Tensor):
            processed["extra.action_is_pad"] = action_is_pad
        return processed


class PatchPolicyPostprocessor(nn.Module):
    """Denormalize actions and apply the execution horizon.

    Args:
        output_features: Action features and their normalization statistics.
        n_action_steps: Number of action timesteps to retain.
    """

    def __init__(
        self,
        output_features: list[Feature],
        n_action_steps: int,
        action_normalization: NormalizationType = NormalizationType.MEAN_STD,
    ) -> None:
        """Initialize action denormalizers and the execution horizon."""
        super().__init__()
        self.output_features = output_features
        if action_normalization is NormalizationType.MEAN_STD:
            _warn_missing_action_normalization(output_features, "de-normalization")
        self.n_action_steps = n_action_steps
        self._action_denormalizers = nn.ModuleList(
            _normalizer(feature, f"feature_{index}", mode=action_normalization, inverse=True)
            for index, feature in enumerate(output_features)
        )

    def forward(self, actions: Tensor) -> Tensor:
        """Denormalize action features and truncate the predicted horizon.

        Args:
            actions: Normalized action chunks.

        Returns:
            Denormalized chunks limited to the execution horizon.

        """
        return _normalize_features(actions, self.output_features, self._action_denormalizers)[:, : self.n_action_steps]


def make_policy_processors(
    config: PatchPolicyConfig,
    image_size: tuple[int, int],
) -> tuple[PatchPolicyPreprocessor, PatchPolicyPostprocessor]:
    """Create matching Patch Policy processors.

    Args:
        config: Resolved policy configuration.
        image_size: Input image height and width resolved by the encoder.

    Returns:
        The input preprocessor and action postprocessor.
    """
    action_normalization = (
        NormalizationType.MIN_MAX if config.action_head == "diffusion" else NormalizationType.IDENTITY
    )
    return (
        PatchPolicyPreprocessor(
            config.input_features,
            config.output_features,
            image_size,
            config.n_obs_steps,
            action_normalization,
        ),
        PatchPolicyPostprocessor(config.output_features, config.n_action_steps, action_normalization),
    )
