# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). The full model / preprocessor pipeline (which loads Qwen3-VL-4B) is
covered separately; these tests exercise the lazy-init path, config wiring,
hyperparameter capture, error handling, and the policy factory.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportBackend
from physicalai.export.backends import TorchExportParameters
from physicalai.inference.data import InferenceFeatureDtype, InferenceFeatureType
from physicalai.policies import get_physicalai_policy_class, get_policy
from physicalai.policies.xr0 import XR0, XR0Config



def _minimal_export_stats() -> dict[str, dict[str, Any]]:
    """Return minimal dataset statistics for exercising the export hooks."""
    return {
        "observation.state": {
            "name": "observation.state",
            "shape": (8,),
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "type": "STATE",
        },
        "observation.images.base": {
            "name": "observation.images.base",
            "shape": (3, 256, 256),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "type": "VISUAL",
        },
        "action": {
            "name": "action",
            "shape": (6,),
            "mean": [0.0] * 6,
            "std": [1.0] * 6,
            "type": "ACTION",
        },
    }


class TestXR0Config:
    """Config resolution through the policy constructor."""

    def test_lazy_initialization(self) -> None:
        policy = XR0()
        assert policy.model is None
        assert policy._preprocessor is None
        assert policy._postprocessor is None
        assert policy._dataset_stats is None

    def test_config_wiring(self) -> None:
        policy = XR0(chunk_size=30, n_action_steps=15, optimizer_lr=1e-4, dtype="float32")
        assert isinstance(policy.config, XR0Config)
        assert policy.config.chunk_size == 30
        assert policy.config.n_action_steps == 15
        assert policy.config.optimizer_lr == 1e-4
        assert policy.config.dtype == "float32"
        assert policy._n_action_steps == 15

    def test_hyperparameters_saved(self) -> None:
        policy = XR0(chunk_size=30, optimizer_lr=1e-4, freeze_vision_encoder=True)
        assert policy.hparams.chunk_size == 30
        assert policy.hparams.optimizer_lr == 1e-4
        assert policy.hparams.freeze_vision_encoder is True
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 30


class TestXR0Policy:
    """Policy behaviour without an initialized model."""

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        policy = XR0()
        obs = Observation(state=torch.randn(1, 8))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(obs)

    def test_eval_forward_dispatches_to_predict(self) -> None:
        policy = XR0().eval()
        obs = Observation(state=torch.randn(1, 8))
        # eval forward routes to predict_action_chunk, which raises without a model
        with pytest.raises(ValueError, match="not initialized"):
            policy(obs)


class TestXR0Features:
    """Explicit input/output feature schema handling (no model download)."""

    def test_requires_both_features(self) -> None:
        state = Feature(name="state", ftype=FeatureType.STATE, shape=(8,))
        with pytest.raises(ValueError, match="both input and output features"):
            XR0(input_features=[state], output_features=None)

    def test_feature_properties_raise_before_init(self) -> None:
        policy = XR0()
        with pytest.raises(ValueError, match="no input features"):
            _ = policy.input_features
        with pytest.raises(ValueError, match="no output features"):
            _ = policy.output_features

    def test_features_stats_roundtrip(self) -> None:
        inputs = [
            Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
            Feature(name="base", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        ]
        outputs = [Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(6,))]

        stats = XR0._features_to_stats(inputs, outputs)
        assert set(stats) == {"observation.state", "observation.base", ACTION}

        recon_inputs, recon_outputs = XR0._stats_to_features(stats)
        assert {f.name for f in recon_inputs} == {"state", "base"}
        assert {f.ftype for f in recon_inputs} == {FeatureType.STATE, FeatureType.VISUAL}
        assert [f.name for f in recon_outputs] == [ACTION]
        assert recon_outputs[0].ftype is FeatureType.ACTION
        assert recon_outputs[0].shape == (6,)


class TestXR0Factory:
    """Policy factory registration."""

    def test_factory_class(self) -> None:
        assert get_physicalai_policy_class("xr0") is XR0

    def test_get_policy(self) -> None:
        policy = get_policy("xr0", chunk_size=30, optimizer_lr=1e-4)
        assert isinstance(policy, XR0)
        assert policy.model is None


class TestXR0Export:
    """Torch export hooks (no model download)."""

    def test_supported_backends_torch_and_openvino(self) -> None:
        assert XR0.get_supported_export_backends() == [ExportBackend.TORCH, ExportBackend.OPENVINO]

    def test_schemas_none_before_init(self) -> None:
        policy = XR0()
        assert policy.inputs_schema is None
        assert policy.outputs_schema is None

    def test_extra_export_args_torch_only(self) -> None:
        policy = XR0()
        extra = policy.extra_export_args
        assert set(extra) == {"torch"}
        assert isinstance(extra["torch"], TorchExportParameters)

    def test_extra_export_args_trims_when_chunk_differs(self) -> None:
        trimmed = XR0(chunk_size=30, n_action_steps=15).extra_export_args["torch"]
        assert any(spec.type == "action_chunk_trimmer" for spec in trimmed.postprocessors_specs)

        untrimmed = XR0(chunk_size=30, n_action_steps=30).extra_export_args["torch"]
        assert all(spec.type != "action_chunk_trimmer" for spec in untrimmed.postprocessors_specs)

    def test_inputs_schema_from_features(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set the features directly rather than via the constructor: passing
        # input/output features to ``XR0(...)`` derives dataset stats from them,
        # which triggers the eager model build (Qwen3-VL-4B download). Here we
        # mimic that reconstructed schema without the build.
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        schema = policy.inputs_schema
        assert schema is not None
        by_name = {feature.name: feature for feature in schema}
        assert set(by_name) == {STATE, IMAGES, TASK}
        assert by_name[STATE].ftype is InferenceFeatureType.STATE
        assert by_name[STATE].shape == (8,)
        assert by_name[IMAGES].ftype is InferenceFeatureType.VISUAL
        assert by_name[TASK].ftype is InferenceFeatureType.LANGUAGE

    def test_outputs_schema_from_features(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set features directly: passing them to ``XR0(...)`` would build dataset
        # stats and trigger the eager model build (Qwen3-VL-4B download).
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        schema = policy.outputs_schema
        assert schema is not None
        assert len(schema) == 1
        assert schema[0].name == ACTION
        assert schema[0].ftype is InferenceFeatureType.ACTION
        assert schema[0].shape == (30, 6)

    def test_multi_camera_inputs_schema_names_views(self) -> None:
        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        # Set features directly: passing them to ``XR0(...)`` would build dataset
        # stats and trigger the eager model build (Qwen3-VL-4B download).
        policy._input_features = [
            Feature(name="state", ftype=FeatureType.STATE, shape=(8,)),
            Feature(name="base", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
            Feature(name="wrist_left", ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
        ]
        policy._output_features = [Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(6,))]

        names = {feature.name for feature in policy.inputs_schema or []}
        # Two cameras -> per-view names (no single-camera IMAGES collapse).
        assert names == {STATE, f"{IMAGES}.base", f"{IMAGES}.wrist_left", TASK}


class TestXR0DeltaMode:
    """Delta vs absolute ``action_mode`` wiring matches across the policy surface.

    Parametrized over both modes so the download-free surface is checked to line up for each.
    The OpenVINO ``extra_export_args`` / ``_bake_ingraph_export`` sides of the
    delta contract need the real model/preprocessor and are covered by the
    export tests; the pass-through toggle itself lives in ``test_export_openvino``.
    """

    @pytest.mark.parametrize("action_mode", ["absolute", "delta"])
    def test_config_action_mode(self, action_mode: str) -> None:
        # The mode round-trips into the resolved config for both values.
        assert XR0(action_mode=action_mode).config.action_mode == action_mode

    @pytest.mark.parametrize(
        ("action_mode", "expected_names"),
        [("absolute", [ACTION]), ("delta", [ACTION, STATE])],
    )
    def test_outputs_schema_matches_mode(self, action_mode: str, expected_names: list[str]) -> None:
        policy = XR0(chunk_size=30, action_mode=action_mode)
        # Sentinel model bypasses the lazy-init guard; delta reads ``state_shape``.
        policy.model = types.SimpleNamespace(state_shape=(1, 8))  # type: ignore[assignment]
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        schema = policy.outputs_schema
        assert schema is not None
        assert [feature.name for feature in schema] == expected_names
        # The action output is identical across modes.
        assert schema[0].ftype is InferenceFeatureType.ACTION
        assert schema[0].shape == (30, 6)
        assert schema[0].dtype is InferenceFeatureDtype.FLOAT32
        # Delta mode adds the current-frame state pass-through as a second output.
        if action_mode == "delta":
            state_out = schema[1]
            assert state_out.ftype is InferenceFeatureType.STATE
            assert state_out.name == STATE
            assert state_out.shape == (1, 8)
            assert state_out.dtype is InferenceFeatureDtype.FLOAT32

    @pytest.mark.parametrize(
        ("mean", "std", "expect_stats"),
        [
            (None, None, False),
            ([[0.0] * 6] * 30, [[1.0] * 6] * 30, True),
        ],
    )
    def test_delta_stats_wiring(
        self,
        mean: list[list[float]] | None,
        std: list[list[float]] | None,
        expect_stats: bool,
    ) -> None:
        policy = XR0(action_mode="delta", action_delta_mean=mean, action_delta_std=std)
        if expect_stats:
            # Stored as float32 tensors for the preprocessor and mirrored into
            # hparams as plain lists so they round-trip through checkpoints.
            assert isinstance(policy._action_delta_mean, torch.Tensor)
            assert isinstance(policy._action_delta_std, torch.Tensor)
            assert policy._action_delta_mean.dtype is torch.float32
            assert policy._action_delta_std.dtype is torch.float32
            assert policy.hparams["action_delta_mean"] == mean
            assert policy.hparams["action_delta_std"] == std
        else:
            assert policy._action_delta_mean is None
            assert policy._action_delta_std is None
            # ``save_hyperparameters`` still captures the init args, but they are
            # left as ``None`` (not overwritten with the mirrored lists).
            assert policy.hparams["action_delta_mean"] is None
            assert policy.hparams["action_delta_std"] is None

