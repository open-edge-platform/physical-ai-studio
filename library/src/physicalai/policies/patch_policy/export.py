# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch and OpenVINO export specifications for Patch Policy."""

from __future__ import annotations

from abc import abstractmethod
from math import prod
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data import FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters

from .processors import resolve_camera_name

if TYPE_CHECKING:
    from .config import PatchPolicyConfig
    from .model import PatchPolicyModel


class PatchPolicyExportMixin(ExportablePolicyMixin):
    """Export checkpoints without changing the shared whole-policy Torch adapter.

    Model-only backend parameters describe the NumPy processing pipeline.
    Torch keeps normalization and action postprocessing inside the policy.
    Observation history is supplied by the caller, not assembled here.
    """

    @property
    @abstractmethod
    def config(self) -> PatchPolicyConfig:
        """The resolved policy configuration."""

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Describe camera preparation, normalization, and action processing in order.

        OpenVINO runs the model with external NumPy processing.
        Torch invokes the whole policy and therefore
        must not apply normalization or action postprocessing a second time.

        Raises:
            ValueError: If an action feature has no concrete shape.
        """
        if self.model is None:
            return {}
        config = self.config
        model = cast("PatchPolicyModel", self.model)

        # Order camera windows, scale pixels to float32, then resize without letterboxing.
        preproc_specs = [
            ComponentSpec(
                type="patch_policy",
                camera_names=[
                    resolve_camera_name(feature)
                    for feature in config.input_features
                    if feature.ftype is FeatureType.VISUAL
                ],
                n_obs_steps=config.n_obs_steps,
            ),
            ComponentSpec(type="to_float_tensor"),
            ComponentSpec(
                type="resize",
                image_resolution=model.encoder.image_size,
                mode="stretch",
            ),
        ]

        # Shape per-camera statistics to broadcast over batch, time, and spatial axes.
        image_stats: dict[str, dict[str, Any]] = {}
        for feature in config.input_features:
            normalization = feature.normalization_data
            if (
                feature.ftype is not FeatureType.VISUAL
                or normalization is None
                or normalization.mean is None
                or normalization.std is None
            ):
                continue
            key = f"images.{resolve_camera_name(feature)}"
            image_stats[key] = {
                name: np.asarray(getattr(normalization, name)).reshape(-1, 1, 1).tolist() for name in ("mean", "std")
            }
            if normalization.mask is not None:
                image_stats[key]["mask"] = normalization.mask

        # Model-only inference also normalizes images; restore float32 after inline JSON stats.
        model_preproc_specs = list(preproc_specs)
        if image_stats:
            model_preproc_specs.extend([
                ComponentSpec(type="normalize", stats=image_stats, mode="mean_std"),
                ComponentSpec(type="to_float_tensor"),
            ])

        # Map adapter-specific output names to the canonical action key.
        postprocessors = [
            ComponentSpec(class_path="physicalai.inference.postprocessors.ActionNormalizer"),
        ]

        # Diffusion predicts normalized actions; concatenate stats in output-feature order.
        # Missing statistics and masked dimensions remain identity; BeT needs no denormalization.
        if config.action_head == "diffusion":
            minimum, maximum, mask = [], [], []
            for feature in config.output_features:
                if feature.shape is None:
                    msg = f"Action feature {feature.name!r} requires a concrete shape for export."
                    raise ValueError(msg)
                width = prod(feature.shape)
                normalization = feature.normalization_data
                if normalization is not None and normalization.min is not None and normalization.max is not None:
                    minimum.extend(np.broadcast_to(normalization.min, (width,)))
                    maximum.extend(np.broadcast_to(normalization.max, (width,)))
                    feature_mask = normalization.mask if normalization.mask is not None else True
                    mask.extend(np.broadcast_to(feature_mask, (width,)))
                else:
                    minimum.extend(np.zeros(width))
                    maximum.extend(np.zeros(width))
                    mask.extend(np.zeros(width, dtype=np.bool_))
            if any(mask):
                postprocessors.append(
                    ComponentSpec(
                        type="denormalize",
                        mode="min_max",
                        stats={
                            "action": {
                                "min": np.asarray(minimum).tolist(),
                                "max": np.asarray(maximum).tolist(),
                                "mask": np.asarray(mask).tolist(),
                            },
                        },
                    ),
                )
        # Execute only the configured prefix of the predicted action chunk.
        if config.n_action_steps != config.chunk_size:
            postprocessors.append(ComponentSpec(type="action_chunk_trimmer", n_action_steps=config.n_action_steps))

        # Torch calls the whole policy; only model-only backends need external normalization/postprocessing.
        return {
            "torch": TorchExportParameters(preprocessors_specs=preproc_specs),
            "openvino": OpenVINOExportParameters(
                inputs=[feature.name for feature in (self.inputs_schema or [])],
                preprocessors_specs=model_preproc_specs,
                postprocessors_specs=postprocessors,
            ),
        }

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe camera windows, excluding the batch axis.

        Raises:
            ValueError: If an image feature has no concrete shape.
        """
        if self.model is None:
            return None
        schema = []
        for feature in self.config.input_features:
            if feature.ftype is not FeatureType.VISUAL:
                continue
            if feature.shape is None:
                msg = f"Image feature {feature.name!r} requires a concrete shape for export."
                raise ValueError(msg)
            schema.append(
                InferenceFeature(
                    name=f"images.{resolve_camera_name(feature)}",
                    ftype=InferenceFeatureType.VISUAL,
                    shape=(self.config.n_obs_steps, *feature.shape),
                    dtype=InferenceFeatureDtype.FLOAT32,
                ),
            )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the physical-unit action chunk returned by Torch."""
        if self.model is None:
            return None
        return [
            InferenceFeature(
                name="action",
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.n_action_steps, self.config.action_dim),
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return the supported checkpoint and model-only backends."""
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
