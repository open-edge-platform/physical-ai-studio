# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 export integration."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast, override

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.observation import (
    ACTION,
    IMAGES,
    STATE,
    TASK,
    Feature,
    FeatureType,
    NormalizationValue,
)
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import ExportParameters, OpenVINOExportParameters, TorchExportParameters
from physicalai.policies.utils.features import get_feature_by_type

from .constants import SO101_JOINT_OFFSETS, SO101_JOINT_SIGNS

if TYPE_CHECKING:
    from .config import MolmoAct2Config
    from .processors import MolmoAct2Preprocessor


def _normalization_stats(
    feature: Feature | None,
) -> dict[str, NormalizationValue | list[bool]]:
    if feature is None or feature.normalization_data is None:
        return {}
    normalization = feature.normalization_data
    stats: dict[str, NormalizationValue | list[bool]] = {
        name: value
        for name in ("mean", "std", "min", "max", "q01", "q99")
        if (value := getattr(normalization, name)) is not None
    }
    if normalization.mask is not None:
        stats["mask"] = normalization.mask
    return stats


class MolmoAct2ExportMixin(ExportablePolicyMixin):
    """Provide MolmoAct2-specific export schemas and backend parameters."""

    input_features: list[Feature] | None
    output_features: list[Feature] | None
    chunk_size: int
    n_action_steps: int
    openvino_compress_to_fp16: bool

    @abstractmethod
    def _require_config(self) -> MolmoAct2Config:
        """Return the initialized MolmoAct2 configuration."""

    @property
    @override
    def sample_input(self) -> dict[str, torch.Tensor | str] | None:
        """A deterministic export sample valid for pass-through state dimensions.

        The synthetic state is zeroed only when its mask contains pass-through dimensions.
        """
        sample = super().sample_input
        state_feature = get_feature_by_type(self.input_features or [], FeatureType.STATE)
        normalization = state_feature.normalization_data if state_feature is not None else None
        if (
            sample is not None
            and state_feature is not None
            and normalization is not None
            and normalization.mask
            and not all(normalization.mask)
        ):
            state = sample.get(str(state_feature.name))
            if torch.is_tensor(state):
                sample[str(state_feature.name)] = torch.zeros_like(state)
        return sample

    @property
    @override
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe raw observation inputs exposed by exported MolmoAct2 policies.

        Raises:
            ValueError: If an input feature has no concrete shape.
        """
        if self.model is None or self.input_features is None:
            return None

        schema: list[InferenceFeature] = []
        for feature in self.input_features:
            if feature.shape is None:
                msg = f"Input feature '{feature.name}' requires a concrete shape for export."
                raise ValueError(msg)
            if feature.ftype == FeatureType.VISUAL:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=tuple(feature.shape),
                        name=f"{IMAGES}.{feature.name}",
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif feature.ftype == FeatureType.STATE:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=tuple(feature.shape),
                        name=str(feature.name),
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )
        return schema

    @property
    @override
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the denormalized action chunk emitted by exported policies.

        Raises:
            ValueError: If the action feature has no concrete shape.
        """
        if self.model is None or self.output_features is None:
            return None
        action_feature = get_feature_by_type(self.output_features, FeatureType.ACTION)
        if action_feature is None or action_feature.shape is None:
            msg = "MolmoAct2 export requires an action feature with a concrete shape."
            raise ValueError(msg)
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.chunk_size, *action_feature.shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    def _openvino_token_ids(self) -> tuple[int, int, list[int]]:
        config = self._require_config()
        required = {
            "image_start_token_id": config.image_start_token_id,
            "image_end_token_id": config.image_end_token_id,
            "image_patch_id": config.image_patch_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            msg = f"MolmoAct2 OpenVINO export requires token IDs: {', '.join(missing)}"
            raise ValueError(msg)
        preprocessor = cast("MolmoAct2Preprocessor | None", self._preprocessor)
        if preprocessor is None:
            msg = "MolmoAct2 preprocessor must be initialized before export."
            raise ValueError(msg)

        tokenizer = preprocessor.tokenizer
        bos_token_id = tokenizer.bos_token_id
        if not isinstance(bos_token_id, int):
            bos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        if not isinstance(bos_token_id, int) or not isinstance(pad_token_id, int):
            msg = "MolmoAct2 tokenizer must define integer BOS/EOS and padding token IDs."
            raise TypeError(msg)

        image_token_ids = [
            token_id
            for token_id in (
                config.image_patch_id,
                config.image_col_id,
                config.image_start_token_id,
                config.low_res_image_start_token_id,
                config.frame_start_token_id,
                config.image_end_token_id,
                config.frame_end_token_id,
                config.image_low_res_id,
            )
            if token_id is not None
        ]
        return bos_token_id, pad_token_id, image_token_ids

    @property
    @override
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build Torch and OpenVINO export parameters.

        Raises:
            ValueError: If export features, token IDs, or processors are unavailable.
        """
        config = self._require_config()
        if self.input_features is None or self.output_features is None:
            msg = "MolmoAct2 export requires initialized input and output features."
            raise ValueError(msg)

        state_feature = get_feature_by_type(self.input_features, FeatureType.STATE)
        action_feature = get_feature_by_type(self.output_features, FeatureType.ACTION)
        if action_feature is None or action_feature.shape is None:
            msg = "MolmoAct2 export requires an action feature with a concrete shape."
            raise ValueError(msg)
        outputs_schema = self.outputs_schema
        if not outputs_schema:
            msg = "MolmoAct2 export requires an output schema."
            raise ValueError(msg)

        bos_token_id, pad_token_id, image_token_ids = self._openvino_token_ids()
        image_size = (
            int(config.image_processor_size["height"]),
            int(config.image_processor_size["width"]),
        )
        preprocessors = [
            ComponentSpec(
                type="molmoact2",
                image_keys=[
                    str(feature.name)
                    for feature in self.input_features
                    if feature.ftype == FeatureType.VISUAL and feature.name
                ],
                state_stats=_normalization_stats(state_feature),
                normalization_mode=config.normalization_mode,
                image_size=image_size,
                num_state_tokens=config.num_state_tokens,
                setup_type=config.setup_type,
                control_mode=config.control_mode,
                add_setup_tokens=config.add_setup_tokens,
                add_control_tokens=config.add_control_tokens,
            ),
            ComponentSpec(
                type="ov_tokenizer",
                artifact="tokenizer.xml",
            ),
            ComponentSpec(
                type="molmoact2_inputs",
                max_action_dim=config.max_action_dim,
                action_dim=int(action_feature.shape[-1]),
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                image_placeholder_token_id=config.image_placeholder_token_id,
                image_start_token_id=config.image_start_token_id,
                image_end_token_id=config.image_end_token_id,
                image_patch_id=config.image_patch_id,
                image_col_id=config.image_col_id,
                low_res_image_start_token_id=config.low_res_image_start_token_id,
                frame_start_token_id=config.frame_start_token_id,
                frame_end_token_id=config.frame_end_token_id,
                image_low_res_id=config.image_low_res_id,
                image_size=image_size,
                patch_size=config.image_processor_patch_size,
                pooling_size=tuple(config.image_processor_pooling_size),
                image_mean=config.image_processor_mean,
                image_std=config.image_processor_std,
                image_crop_mode=config.image_processor_crop_mode,
                image_use_col_tokens=config.image_use_col_tokens,
                use_single_crop_col_tokens=config.use_single_crop_col_tokens,
                use_single_crop_start_token=config.use_single_crop_start_token,
                image_token_ids=image_token_ids,
            ),
        ]
        if config.adapt_to_so101:
            preprocessors.insert(
                0,
                ComponentSpec(
                    type="joint_frame_preprocess",
                    feature=STATE,
                    signs=list(SO101_JOINT_SIGNS),
                    offsets=list(SO101_JOINT_OFFSETS),
                ),
            )
        torch_postprocessors = []
        openvino_postprocessors = [
            ComponentSpec(
                type="molmoact2_postprocess",
                action_key=outputs_schema[0].name,
                action_stats=_normalization_stats(action_feature),
                normalization_mode=config.normalization_mode,
            ),
        ]
        if config.adapt_to_so101:
            openvino_postprocessors.append(
                ComponentSpec(
                    type="joint_frame_postprocess",
                    feature=ACTION,
                    signs=list(SO101_JOINT_SIGNS),
                    offsets=list(SO101_JOINT_OFFSETS),
                ),
            )
        if self.chunk_size != self.n_action_steps:
            chunk_trimmer = ComponentSpec(
                type="action_chunk_trimmer",
                n_action_steps=self.n_action_steps,
            )
            torch_postprocessors.append(chunk_trimmer)
            openvino_postprocessors.append(chunk_trimmer)
        return {
            ExportBackend.TORCH: TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=torch_postprocessors,
            ),
            ExportBackend.OPENVINO: OpenVINOExportParameters(
                outputs=[feature.name for feature in outputs_schema],
                export_tokenizer=True,
                compress_to_fp16=self.openvino_compress_to_fp16,
                via_onnx=False,
                preprocessors_specs=preprocessors,
                postprocessors_specs=openvino_postprocessors,
            ),
        }

    @staticmethod
    @override
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Return export backends implemented by MolmoAct2."""
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]
