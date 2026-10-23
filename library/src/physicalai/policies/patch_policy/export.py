# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export contract for Patch Policy."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES
from physicalai.export import ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters

from .processors import resolve_camera_name

if TYPE_CHECKING:
    import torch

    from .config import PatchPolicyConfig


def _mean_std(feature: Feature, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened mean, standard deviation, and normalization mask.

    Raises:
        ValueError: If configured statistics do not match the feature width.
    """
    stats = feature.normalization_data
    if stats is None or stats.mean is None or stats.std is None:
        return np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32), np.zeros(width, dtype=np.bool_)

    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1)
    if mean.size != width or std.size != width:
        msg = f"Feature {feature.name!r} normalization statistics must contain {width} values."
        raise ValueError(msg)
    mask = np.ones(width, dtype=np.bool_)
    if stats.mask is not None:
        mask = np.asarray(stats.mask, dtype=np.bool_).reshape(-1)
        if mask.size != width:
            msg = f"Feature {feature.name!r} normalization mask must contain {width} values."
            raise ValueError(msg)
    return mean, std, mask


def image_normalization_stats(features: list[Feature]) -> dict[str, dict[str, list]] | None:
    """Build broadcastable Runtime statistics for stacked camera images.

    Returns:
        Nested Runtime statistics, or ``None`` when all cameras use identity normalization.
    """
    if not features:
        return None
    channels = features[0].shape[0] if features[0].shape else 0
    values = [_mean_std(feature, channels) for feature in features]
    if not any(mask.any() for _, _, mask in values):
        return None
    means, stds, masks = (np.stack(items) for items in zip(*values, strict=True))
    shape = (len(features), channels, 1, 1)
    return {
        IMAGES: {
            "mean": means.reshape(shape).tolist(),
            "std": stds.reshape(shape).tolist(),
            "mask": masks.reshape(shape).tolist(),
        },
    }


def action_normalization_stats(features: list[Feature]) -> dict[str, dict[str, list]] | None:
    """Build Runtime statistics for the concatenated action vector.

    Returns:
        Nested Runtime statistics, or ``None`` when all actions use identity normalization.
    """
    values = [_mean_std(feature, prod(feature.shape or ())) for feature in features]
    if not values or not any(mask.any() for _, _, mask in values):
        return None
    means, stds, masks = (np.concatenate(items) for items in zip(*values, strict=True))
    return {ACTION: {"mean": means.tolist(), "std": stds.tolist(), "mask": masks.tolist()}}


class PatchPolicyExportMixin:
    """Declarative Torch and OpenVINO export support for Patch Policy."""

    model: torch.nn.Module | None

    @property
    def config(self) -> PatchPolicyConfig:
        """Resolved Patch Policy configuration."""
        raise NotImplementedError

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return the initially supported Patch Policy export backends."""
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]

    def _validated_visual_features(self) -> list[Feature]:
        """Validate and return visual features in configured camera order.

        Returns:
            Ordered visual features.

        Raises:
            ValueError: If the policy lies outside the initial export contract.
        """
        config = self.config
        if config.encoder_name != "dinov2" or config.encoder_output != "patches":
            msg = "Patch Policy export currently supports only encoder_name='dinov2' with encoder_output='patches'."
            raise ValueError(msg)
        features = [feature for feature in config.input_features if feature.ftype is FeatureType.VISUAL]
        names = [resolve_camera_name(feature) for feature in features]
        if len(names) != len(set(names)):
            msg = f"Patch Policy export camera names must be unique, got {names!r}."
            raise ValueError(msg)
        visual_dimensions = 3
        first_shape = features[0].shape
        if first_shape is None:
            msg = f"Visual feature {features[0].name!r} must have a concrete (C, H, W) shape for export."
            raise ValueError(msg)
        channels = first_shape[0]
        for feature in features:
            if feature.shape is None or len(feature.shape) != visual_dimensions:
                msg = f"Visual feature {feature.name!r} must have a concrete (C, H, W) shape for export."
                raise ValueError(msg)
            if feature.shape[0] != channels:
                msg = "All Patch Policy cameras must have the same channel count for export."
                raise ValueError(msg)
        return features

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe raw camera inputs consumed by Runtime preprocessing."""
        if self.model is None:
            return None
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.VISUAL,
                shape=(self.config.n_obs_steps, *feature.shape) if self.config.n_obs_steps > 1 else feature.shape,
                name=f"{IMAGES}.{resolve_camera_name(feature)}",
                dtype=InferenceFeatureDtype.FLOAT32,
            )
            for feature in self._validated_visual_features()
            if feature.shape is not None
        ]

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the normalized full action chunk emitted by the exported graph."""
        if self.model is None:
            return None
        self._validated_visual_features()
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.chunk_size, self.config.action_dim),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build backend-specific Runtime processor manifests."""
        visual_features = self._validated_visual_features()
        image_stats = image_normalization_stats(visual_features)
        action_stats = action_normalization_stats(self.config.output_features)

        openvino_preprocessors = [
            ComponentSpec(
                type="patch_policy",
                camera_names=[resolve_camera_name(feature) for feature in visual_features],
                image_resolution=self.config.image_size,
                n_obs_steps=self.config.n_obs_steps,
            ),
        ]
        if image_stats is not None:
            openvino_preprocessors.append(
                ComponentSpec(type="normalize", stats=image_stats, mode="mean_std", features=[IMAGES]),
            )
        openvino_preprocessors.append(ComponentSpec(type="to_float_tensor"))

        openvino_postprocessors = []
        if action_stats is not None:
            openvino_postprocessors.append(
                ComponentSpec(type="denormalize", stats=action_stats, mode="mean_std", features=[ACTION]),
            )
        if self.config.n_action_steps != self.config.chunk_size:
            openvino_postprocessors.append(
                ComponentSpec(type="action_chunk_trimmer", n_action_steps=self.config.n_action_steps),
            )

        return {
            "torch": TorchExportParameters(
                output_names=[ACTION],
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            ),
            "openvino": OpenVINOExportParameters(
                inputs=[IMAGES],
                outputs=[ACTION],
                export_tokenizer=False,
                compress_to_fp16=False,
                via_onnx=False,
                preprocessors_specs=openvino_preprocessors,
                postprocessors_specs=openvino_postprocessors,
            ),
        }


__all__ = [
    "PatchPolicyExportMixin",
    "action_normalization_stats",
    "image_normalization_stats",
]
