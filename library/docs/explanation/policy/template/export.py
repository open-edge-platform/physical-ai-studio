# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export schema and backend parameters for the native policy design."""

from __future__ import annotations

from physicalai.data import FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import (
    ExecuTorchExportParameters,
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from .config import NewPolicyModelConfig


class NewPolicyExportMixin(ExportablePolicyMixin):  # type: ignore[misc]
    """Export schema and backend parameters, derived from the resolved model config.

    Requires the concrete class to provide `_config: NewPolicyModelConfig | None` and
    `_config_available: bool` (satisfied by `TemplatePolicy`/`NewPolicy`).
    """

    _config: NewPolicyModelConfig | None
    _config_available: bool

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe ACT-style state/image inputs with appended language metadata."""
        if not self._config_available:
            return None
        assert self._config is not None

        config = self._config
        state_features = [feature for feature in config.input_features if feature.ftype == FeatureType.STATE]
        if len(state_features) != 1 or state_features[0].shape is None:
            raise ValueError("Export requires exactly one state feature with a concrete shape")

        schema = [
            InferenceFeature(
                ftype=InferenceFeatureType.STATE,
                shape=tuple(state_features[0].shape),
                name=STATE,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
        ]

        image_features = [feature for feature in config.input_features if feature.ftype == FeatureType.VISUAL]
        for feature in image_features:
            if feature.name is None or feature.shape is None:
                raise ValueError("Export image features must define a name and shape")
            image_name = IMAGES if len(image_features) == 1 else f"{IMAGES}.{feature.name}"
            schema.append(
                InferenceFeature(
                    ftype=InferenceFeatureType.VISUAL,
                    shape=tuple(feature.shape),
                    name=image_name,
                    dtype=InferenceFeatureDtype.FLOAT32,
                )
            )

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(config.tokenizer_max_length,),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            )
        )
        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the ACT-style action chunk produced by the model."""
        if not self._config_available:
            return None
        assert self._config is not None

        config = self._config
        if len(config.output_features) != 1 or config.output_features[0].shape is None:
            raise ValueError("Export requires exactly one action feature with a concrete shape")
        action_feature = config.output_features[0]
        action_shape = action_feature.shape
        if action_shape is None:
            raise ValueError("Export action feature must define a concrete shape")
        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(config.chunk_size, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            )
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Build ACT-style backend parameters from the resolved model config."""
        assert self._config is not None
        config = self._config
        output_names = [feature.name for feature in (self.outputs_schema or [])]
        postprocessors: list[ComponentSpec] = []
        if config.chunk_size != config.n_action_steps:
            postprocessors.append(
                ComponentSpec.model_validate(
                    {
                        "type": "action_chunk_trimmer",
                        "n_action_steps": config.n_action_steps,
                    }
                )
            )

        preprocessors = [
            ComponentSpec.model_validate(
                {
                    "type": "resize",
                    "image_resolution": config.image_size,
                    "mode": "letterbox",
                }
            )
        ]
        return {
            "onnx": ONNXExportParameters(
                exporter_kwargs={"output_names": output_names},
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "openvino": OpenVINOExportParameters(
                outputs=output_names,
                export_tokenizer=False,
                compress_to_fp16=True,
                exporter_kwargs={},
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "executorch": ExecuTorchExportParameters(
                preprocessors_specs=preprocessors,
                postprocessors_specs=postprocessors,
            ),
            "torch": TorchExportParameters(
                preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
                postprocessors_specs=postprocessors,
            ),
        }

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        return [
            ExportBackend.TORCH,
            ExportBackend.OPENVINO,
            ExportBackend.ONNX,
            ExportBackend.EXECUTORCH,
        ]
