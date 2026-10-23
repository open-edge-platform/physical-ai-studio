# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Input and output processing for Patch Policy."""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.data import Feature, FeatureType
from physicalai.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType

if TYPE_CHECKING:
    from .config import PatchPolicyConfig


_NORM_MAP = {
    FeatureType.VISUAL: NormalizationType.MEAN_STD,
    FeatureType.ACTION: NormalizationType.MEAN_STD,
}


def resolve_camera_name(feature: Feature) -> str:
    """Return the canonical camera name for a visual feature.

    Raises:
        ValueError: If the feature has no usable name.
    """
    name = (feature.name or "").removeprefix("observation.images.").removeprefix("images.")
    if not name:
        raise ValueError("Patch Policy visual features must have non-empty camera names.")
    return name


def _normalizable_features(features: list[Feature]) -> dict[str, Feature]:
    """Return indexed features with complete mean/std statistics."""
    return {
        f"feature_{index}": feature
        for index, feature in enumerate(features)
        if feature.normalization_data is not None
        and feature.normalization_data.mean is not None
        and feature.normalization_data.std is not None
    }


def _split_features(values: Tensor, features: list[Feature]) -> dict[str, Tensor]:
    """Split a concatenated tensor into stable, indexed feature entries.

    Raises:
        ValueError: If a feature shape is missing or the total width does not match.
    """
    pieces: dict[str, Tensor] = {}
    offset = 0
    for index, feature in enumerate(features):
        if not feature.shape:
            raise ValueError(f"Feature {feature.name!r} has no shape.")
        width = prod(feature.shape)
        pieces[f"feature_{index}"] = values[..., offset : offset + width]
        offset += width
    if values.shape[-1] != offset:
        raise ValueError(f"Feature width {values.shape[-1]} does not match configured width {offset}.")
    return pieces


def _join_features(values: dict[str, Tensor], features: list[Feature]) -> Tensor:
    """Join indexed feature entries in configuration order."""
    return torch.cat([values[f"feature_{index}"] for index in range(len(features))], dim=-1)


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
    ) -> None:
        super().__init__()
        self.visual_features = [feature for feature in input_features if feature.ftype is FeatureType.VISUAL]
        self.output_features = output_features
        self.image_size = image_size
        self.n_obs_steps = n_obs_steps
        self._image_normalizer = FeatureNormalizeTransform(_normalizable_features(self.visual_features), _NORM_MAP)
        self._action_normalizer = FeatureNormalizeTransform(_normalizable_features(output_features), _NORM_MAP)

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
            if len(images.shape) < 4:
                raise ValueError("Visual input must include batch, channel, height, and width dimensions.")
            return images
        canonical_name = resolve_camera_name(feature)
        names = [feature.name or "", canonical_name, f"images.{canonical_name}"]
        for name in names:
            if name in images and isinstance(images[name], Tensor):
                return images[name]
        raise KeyError(f"Missing visual feature {feature.name!r} in Observation.images.")

    def normalize_actions(self, actions: Tensor) -> Tensor:
        """Normalize each action feature with its configured statistics.

        Args:
            actions: Concatenated action tensor.

        Returns:
            The normalized action tensor.

        """
        pieces = _split_features(actions, self.output_features)
        return _join_features(self._action_normalizer(pieces), self.output_features)

    def forward(self, batch: Mapping[str, Any], *, require_actions: bool = False) -> dict[str, Tensor]:
        """Convert an observation mapping into model inputs.

        Args:
            batch: Observation values containing images and optional actions.
            require_actions: Whether actions must be present and normalized.

        Returns:
            Model inputs containing stacked images and optional action metadata.

        Raises:
            TypeError: If images or required actions are not tensors in a supported layout.
            ValueError: If an image has an unsupported shape.
        """
        raw_images = batch.get("images")
        if raw_images is None:
            raw_images = {
                key.removeprefix("images."): value
                for key, value in batch.items()
                if key.startswith("images.") and isinstance(value, Tensor)
            }
        if not isinstance(raw_images, (Mapping, Tensor)):
            raise TypeError("Observation.images must be a tensor or camera dictionary.")
        views = []
        for index, feature in enumerate(self.visual_features):
            image = self._image(raw_images, feature).float()
            if image.ndim == 4:
                image = image[:, None]
            if image.ndim != 5:
                raise ValueError(f"Expected image shape [B, T, C, H, W], got {tuple(image.shape)}.")
            if image.shape[1] != self.n_obs_steps:
                raise ValueError(
                    f"Expected {self.n_obs_steps} observation timestep(s), got {image.shape[1]} for {feature.name!r}.",
                )
            flat = image.flatten(0, 1)
            if flat.shape[-2:] != self.image_size:
                flat = F.interpolate(flat, size=self.image_size, mode="bilinear", align_corners=False)
            image = flat.reshape(*image.shape[:-2], *self.image_size)
            views.append(self._image_normalizer({f"feature_{index}": image})[f"feature_{index}"])
        processed = {"images": torch.stack(views, dim=2)}

        action = batch.get("action")
        if require_actions:
            if not isinstance(action, Tensor):
                raise TypeError("Training requires tensor actions.")
            processed["action"] = self.normalize_actions(action)
        extra = batch.get("extra")
        if isinstance(extra, Mapping) and isinstance(extra.get("action_is_pad"), Tensor):
            processed["extra.action_is_pad"] = extra["action_is_pad"]
        return processed


class PatchPolicyPostprocessor(nn.Module):
    """Denormalize actions and apply the execution horizon.

    Args:
        output_features: Action features and their normalization statistics.
        n_action_steps: Number of action timesteps to retain.
    """

    def __init__(self, output_features: list[Feature], n_action_steps: int) -> None:
        super().__init__()
        self.output_features = output_features
        self.n_action_steps = n_action_steps
        self._action_denormalizer = FeatureNormalizeTransform(
            _normalizable_features(output_features),
            _NORM_MAP,
            inverse=True,
        )

    def forward(self, actions: Tensor) -> Tensor:
        """Denormalize action features and truncate the predicted horizon.

        Args:
            actions: Normalized action chunks.

        Returns:
            Denormalized chunks limited to the execution horizon.

        """
        pieces = _split_features(actions, self.output_features)
        denormalized = self._action_denormalizer(pieces)
        return _join_features(denormalized, self.output_features)[:, : self.n_action_steps]


def make_policy_processors(config: PatchPolicyConfig) -> tuple[PatchPolicyPreprocessor, PatchPolicyPostprocessor]:
    """Create matching Patch Policy processors.

    Args:
        config: Resolved policy configuration.

    Returns:
        The input preprocessor and action postprocessor.
    """
    return (
        PatchPolicyPreprocessor(config.input_features, config.output_features, config.image_size, config.n_obs_steps),
        PatchPolicyPostprocessor(config.output_features, config.n_action_steps),
    )
