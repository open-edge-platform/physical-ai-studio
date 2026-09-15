# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""RLDX-1 export mixin.

This mixin layers RLDX-1-specific export behavior on top of
``ExportablePolicyMixin`` while delegating shared export mechanics to the base
mixin implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import (
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.policies.rldx1.components.backbone.graph_safe_rldx1 import GraphSafeRldx1Model
from physicalai.policies.rldx1.utils.export import (
    build_compress_reference_ids,
    build_padded_sample,
    build_rldx1_token_composer_params,
    cast_sample_fp32,
    export_image_resolution_from_stats,
    fp32_weights_for_export,
    trim_export_sample,
)
from physicalai.policies.rldx1.utils.stats import get_dataset_stats_entry, resolve_feature_shape

from .constants import ATTENTION_MASK, INPUT_IDS, PIXEL_VALUES, POSITION_IDS
from .vtc_buffer import VtcWindowBuffer

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

    from physicalai.policies.rldx1.config import Rldx1Config
    from physicalai.policies.rldx1.model import Rldx1Model

    from .preprocessor import Rldx1Preprocessor

DatasetStats = dict[str, dict[str, Any]]


class Rldx1ExportMixin(ExportablePolicyMixin):
    """RLDX-1-specific export behavior layered on ExportablePolicyMixin."""

    # Structural typing for the concrete owner policy (Rldx1).
    config: Rldx1Config
    model: torch.nn.Module
    _preprocessor: torch.nn.Module
    _dataset_stats: DatasetStats | None
    _camera_names: list[str]

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        Returns:
            list[str | ExportBackend]: Supported export backend identifiers.
        """
        return [ExportBackend.TORCH, ExportBackend.ONNX, ExportBackend.OPENVINO]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Returns:
            list[InferenceFeature] | None: Input schema for export tracing, or
                ``None`` when the model or dataset statistics are unavailable.

        Raises:
            ValueError: If dataset statistics have no visual features.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        dataset_stats = self._dataset_stats

        schema: list[InferenceFeature] = []

        num_image_features = sum(
            1 for feature in dataset_stats.values() if str(FeatureType.VISUAL) in str(feature.get("type", ""))
        )
        for feature_id, feature in dataset_stats.items():
            feature_type = str(feature.get("type", ""))
            if STATE in feature_id:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=resolve_feature_shape(feature),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif str(FeatureType.VISUAL) in feature_type:
                feature_name = (
                    str(feature.get("name", feature_id)).removeprefix("observation.").removeprefix(f"{IMAGES}.")
                )
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=resolve_feature_shape(feature),
                        name=name,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        if num_image_features == 0:
            msg = (
                "dataset_stats carries no visual features. Pass input_features={'<view>': "
                "Feature(ftype=FeatureType.VISUAL, shape=(3, height, width)), ...} to "
                f"Rldx1(...) to export this policy. Camera names discovered from "
                f"processor_config.json: {self._camera_names or '(none found)'}."
            )
            raise ValueError(msg)

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
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export."""
        if self.model is None or self._dataset_stats is None:
            return None

        action_shape = resolve_feature_shape(get_dataset_stats_entry(self._dataset_stats, ACTION))

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.action_horizon, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        Returns:
            dict[str, ExportParameters]: Backend-specific export parameters.

        Raises:
            RuntimeError: If the preprocessor is not initialized.
            ValueError: If dataset statistics are missing or contain no visual
                features needed to derive export image resolution.
        """
        if self._dataset_stats is None:
            msg = (
                "Dataset stats are required for export. Initialize the policy with dataset_stats"
                " or train for at least one epoch to populate them."
            )
            raise ValueError(msg)

        try:
            image_resolution = export_image_resolution_from_stats(self._dataset_stats)
        except RuntimeError as exc:
            msg = (
                "dataset_stats carries no visual features. Pass input_features={'<view>': "
                "Feature(ftype=FeatureType.VISUAL, shape=(3, height, width)), ...} to "
                "Rldx1(...) to export this policy."
            )
            raise ValueError(msg) from exc

        normalize_spec = ComponentSpec(
            type="normalize",
            stats={STATE: get_dataset_stats_entry(self._dataset_stats, f"observation.{STATE}", STATE)},
            mode="quantiles",
        )
        postproc_specs = [
            ComponentSpec(
                type="denormalize",
                stats={ACTION: self._dataset_stats[ACTION]},
                mode="quantiles",
            ),
        ]
        if self._preprocessor is None:
            msg = "Cannot build token composer params before transforms are initialized."
            raise RuntimeError(msg)
        preprocessor = cast("Rldx1Preprocessor", self._preprocessor)
        token_composer_params = build_rldx1_token_composer_params(
            tokenizer=preprocessor.tokenizer,
            image_resolution=image_resolution,
            num_views=int(self.config.num_views or 1),
            num_frames=int(self.config.video_length),
            max_token_len=int(self.config.tokenizer_max_length),
        )

        rope_specs: list[ComponentSpec] = []
        if self.model is not None:
            backbone = getattr(self.model, "backbone", None) or getattr(self.model, "gs_backbone", None)
            if backbone is not None:
                qwen_config = backbone.qwen_config
                rope_specs = [
                    ComponentSpec(
                        type="rldx1_rope",
                        image_token_id=qwen_config.image_token_id,
                        vision_start_token_id=qwen_config.vision_start_token_id,
                        spatial_merge_size=qwen_config.vision_config.spatial_merge_size,
                        n_cog_tokens=self.config.n_cog_tokens,
                    ),
                ]

        extra_args: dict[str, ExportParameters] = {}
        num_views = int(self.config.num_views or 1)
        num_frames = int(self.config.video_length)
        output_names = [feature.name for feature in (self.outputs_schema or [])]
        extra_args["onnx"] = ONNXExportParameters(
            exporter_kwargs={
                "output_names": output_names,
            },
            export_tokenizer=False,
            preprocessors_specs=[
                normalize_spec,
                ComponentSpec(
                    type="rldx1",
                    image_resolution=image_resolution,
                    num_views=num_views,
                    num_frames=num_frames,
                    max_state_dim=self.config.max_state_dim,
                ),
                ComponentSpec(
                    type="hf_tokenizer",
                    tokenizer_name="RLWRLD/RLDX-1-VLM",
                    max_token_len=self.config.tokenizer_max_length,
                ),
                ComponentSpec(
                    type="rldx1_token_composer",
                    **token_composer_params,
                ),
                *rope_specs,
            ],
            postprocessors_specs=postproc_specs,
        )
        extra_args["openvino"] = OpenVINOExportParameters(
            inputs=[PIXEL_VALUES, INPUT_IDS, POSITION_IDS, ATTENTION_MASK, STATE],
            outputs=output_names,
            compress_to_fp16=self.config.compress_to_fp16,
            via_onnx=False,
            export_tokenizer=True,
            exporter_kwargs={},
            preprocessors_specs=[
                normalize_spec,
                ComponentSpec(
                    type="rldx1",
                    image_resolution=image_resolution,
                    num_views=num_views,
                    num_frames=num_frames,
                    max_state_dim=self.config.max_state_dim,
                ),
                ComponentSpec(
                    type="ov_tokenizer",
                    artifact="tokenizer.xml",
                ),
                ComponentSpec(
                    type="rldx1_token_composer",
                    **token_composer_params,
                ),
                *rope_specs,
            ],
            postprocessors_specs=postproc_specs,
        )
        extra_args["torch"] = TorchExportParameters(
            preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            postprocessors_specs=[],
        )

        return extra_args

    def _build_graph_safe_model(
        self,
        model: Rldx1Model,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_views: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> GraphSafeRldx1Model:
        """Build the export-only graph-safe view over a trained model.

        Returns:
            GraphSafeRldx1Model: Graph-safe wrapper used during export tracing.

        Raises:
            RuntimeError: If preprocessor or dataset statistics are not
                available when constructing export-only state.
        """
        outputs_schema = self.outputs_schema or []
        action_dim = self.config.max_action_dim
        if outputs_schema and outputs_schema[0].shape:
            action_dim = int(outputs_schema[0].shape[-1])

        if self._preprocessor is None or self._dataset_stats is None:
            msg = "No preprocessor available to build compress reference ids."
            raise RuntimeError(msg)
        preprocessor = cast("Rldx1Preprocessor", self._preprocessor)
        image_resolution = export_image_resolution_from_stats(self._dataset_stats)
        token_composer_params = build_rldx1_token_composer_params(
            tokenizer=preprocessor.tokenizer,
            image_resolution=image_resolution,
            num_views=int(self.config.num_views or 1),
            num_frames=int(self.config.video_length),
            max_token_len=int(self.config.tokenizer_max_length),
        )
        compress_input_ids = build_compress_reference_ids(token_composer_params)
        return GraphSafeRldx1Model(
            model,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            num_views=num_views,
            config=self.config,
            output_action_dim=action_dim,
            embodiment_id=embodiment_id,
            compress_input_ids=compress_input_ids,
        )

    @contextmanager
    def _graph_safe_export_model(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        embodiment_id: torch.Tensor,
        num_views: torch.Tensor,
    ) -> Generator[None, None, None]:
        """Temporarily swap self.model for its graph-safe export view.

        Yields:
            None: Control within a context where ``self.model`` is graph-safe.

        Raises:
            RuntimeError: If export is requested before model initialization.
        """
        if self.model is None:
            msg = "Cannot export before the model is initialized (call setup / load a checkpoint first)."
            raise RuntimeError(msg)

        original = cast("Rldx1Model", self.model)

        graph_safe = self._build_graph_safe_model(
            original,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            num_views=num_views,
            embodiment_id=embodiment_id,
        )
        try:
            self.model = graph_safe  # type: ignore[assignment]
            yield
        finally:
            if hasattr(graph_safe, "restore"):
                graph_safe.restore()
            self.model = original

    @torch.no_grad()
    def to_onnx(  # pyrefly: ignore[bad-override, bad-override-param-name]
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict[str, object],
    ) -> None:
        """Export to ONNX using graph-safe tracing."""
        if input_sample is None:
            input_sample = self._get_default_export_input_sample()
        sample_tensors = input_sample
        with (
            fp32_weights_for_export(self.model),
            self._graph_safe_export_model(
                input_ids=sample_tensors["input_ids"],
                image_grid_thw=sample_tensors["image_grid_thw"],
                embodiment_id=sample_tensors["embodiment_id"],
                num_views=sample_tensors["num_views"],
            ),
        ):
            traced_sample = cast_sample_fp32(sample_tensors)
            trimmed_sample = trim_export_sample(traced_sample)
            super().to_onnx(output_path, input_sample=trimmed_sample, **export_kwargs)

    @torch.no_grad()
    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: object,
    ) -> None:
        """Export to OpenVINO using graph-safe tracing."""
        if input_sample is None:
            input_sample = self._get_default_export_input_sample()
        with (
            fp32_weights_for_export(self.model),
            self._graph_safe_export_model(
                input_ids=input_sample["input_ids"],
                image_grid_thw=input_sample["image_grid_thw"],
                embodiment_id=input_sample["embodiment_id"],
                num_views=input_sample["num_views"],
            ),
        ):
            traced_sample = cast_sample_fp32(input_sample)
            trimmed_sample = trim_export_sample(traced_sample)
            base = cast("Any", super())
            base.to_openvino(output_path, input_sample=trimmed_sample, **export_kwargs)

    @torch.no_grad()
    def _get_default_export_input_sample(self) -> dict[str, torch.Tensor]:
        """Build the default export sample using VTC-prepared model input.

        Returns:
            dict[str, torch.Tensor]: Tensor-only sample ready for export.

        Raises:
            RuntimeError: If sample input or preprocessor is unavailable.
        """
        sample = self.sample_input
        if sample is None:
            msg = "No sample input available for export."
            raise RuntimeError(msg)
        sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

        export_vtc = VtcWindowBuffer(
            video_length=int(self.config.video_length),
            video_stride=int(self.config.video_stride),
        )
        model_input = export_vtc.prepare(sample)

        if self._preprocessor is None:
            msg = "No preprocessor available to build export sample."
            raise RuntimeError(msg)
        preprocessor = cast("Rldx1Preprocessor", self._preprocessor)
        processed_sample = preprocessor(model_input)
        tensor_sample = {k: v for k, v in processed_sample.items() if isinstance(v, torch.Tensor)}

        if self.model is None:
            return tensor_sample

        padded_sample = build_padded_sample(
            cast("Rldx1Model", self.model),
            input_ids=tensor_sample["input_ids"],
            image_grid_thw=tensor_sample["image_grid_thw"],
            embodiment_id=tensor_sample["embodiment_id"],
            config=self.config,
        )
        tensor_sample.update(padded_sample)
        return tensor_sample
