# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Lightning policy wrapper.

Fast, self-contained tests with no external dependencies (no HuggingFace model
downloads). The full model / preprocessor pipeline (which loads Qwen3-VL-4B) is
covered separately; these tests exercise the lazy-init path, config wiring,
hyperparameter capture, error handling, and the policy factory.
"""

from __future__ import annotations

import sys
import types
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import lightning.pytorch as L
import pytest
import torch

from physicalai.data import Feature, FeatureType, Observation
from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK
from physicalai.export import ExportBackend
from physicalai.export.backends import TorchExportParameters
from physicalai.inference.data import InferenceFeatureDtype, InferenceFeatureType
from physicalai.policies import get_physicalai_policy_class, get_policy
from physicalai.policies.xr0 import XR0, XR0Config, XR0Model, make_xr0_preprocessors
from physicalai.policies.xr0.policy import _build_optimizer  # noqa: PLC2701


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

    def test_default_attention_backend(self) -> None:
        assert XR0Config().vlm_attn_implementation == "sdpa"
        assert signature(XR0Model).parameters["vlm_attn_implementation"].default == "sdpa"
        policy = XR0()
        assert policy.config.vlm_attn_implementation == "sdpa"
        assert policy.hparams["vlm_attn_implementation"] == "sdpa"
        assert XR0(vlm_attn_implementation="flash_attention_2").config.vlm_attn_implementation == "flash_attention_2"

    def test_lazy_initialization(self) -> None:
        policy = XR0()
        assert policy.config.dtype == XR0Config().dtype == "float32"
        assert policy.hparams["dtype"] == "float32"
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

    def test_image_key_view_map_defaults_empty(self) -> None:
        assert XR0().config.image_key_view_map == {}

    def test_image_key_view_map_wiring(self) -> None:
        view_map = {"images.top_camera": "ego", "images.pov_black_follower_camera": "wrist_left"}
        policy = XR0(image_key_view_map=view_map)
        assert policy.config.image_key_view_map == view_map
        assert policy.hparams.image_key_view_map == view_map


def test_build_optimizer_adamw() -> None:
    """Plain AdamW keeps parameter groups and config hyperparameters."""
    param = torch.nn.Parameter(torch.ones(1))
    config = XR0Config(optimizer_type="adamw", optimizer_lr=2e-4, optimizer_betas=(0.8, 0.9), optimizer_eps=1e-7)
    optimizer = _build_optimizer(config, [{"params": [param], "weight_decay": 0.02}])
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["params"] == [param]
    assert optimizer.param_groups[0]["weight_decay"] == 0.02
    assert (optimizer.defaults["lr"], optimizer.defaults["betas"], optimizer.defaults["eps"]) == (2e-4, (0.8, 0.9), 1e-7)


@pytest.mark.parametrize(("optimizer_type", "dtype"), [("adamw4bit", "float32"), ("adamw8bit", "bfloat16")])
def test_build_optimizer_low_bit(optimizer_type: str, dtype: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Choose the low-bit variant and enable stochastic rounding for BF16."""
    adamw4, adamw8 = MagicMock(), MagicMock()
    optim = types.ModuleType("torchao.optim")
    optim.__dict__.update(AdamW4bit=adamw4, AdamW8bit=adamw8)
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.optim", optim)
    groups = [{"params": [torch.nn.Parameter(torch.ones(1))], "weight_decay": 0.02}]
    config = XR0Config(optimizer_type=optimizer_type, dtype=dtype, optimizer_lr=2e-4)

    optimizer = _build_optimizer(config, groups)

    chosen = adamw4 if optimizer_type == "adamw4bit" else adamw8
    chosen.assert_called_once_with(
        groups, lr=2e-4, betas=config.optimizer_betas, eps=config.optimizer_eps,
        bf16_stochastic_round=dtype == "bfloat16",
    )
    assert optimizer is chosen.return_value


def test_build_optimizer_falls_back_without_torchao(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Low-bit optimizers fall back to plain AdamW with a warning when torchao is missing."""
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setitem(sys.modules, "torchao.optim", None)
    with caplog.at_level("WARNING"):
        optimizer = _build_optimizer(XR0Config(), [{"params": [torch.nn.Parameter(torch.ones(1))]}])
    assert type(optimizer) is torch.optim.AdamW
    assert "requires the 'torchao' package" in caplog.text


class TestXR0Policy:
    """Policy behaviour without an initialized model."""

    def test_augmentation_only_on_training_forward(self) -> None:
        policy = XR0(augment_images=True)
        policy._preprocessor = MagicMock(return_value={})  # noqa: SLF001
        policy.model = MagicMock()
        obs = Observation(state=torch.randn(1, 8))

        policy.forward(obs)
        policy.compute_val_loss(obs)

        assert policy.hparams.augment_images is True
        assert policy._preprocessor.call_args_list[0].kwargs == {"augment_images": True}  # noqa: SLF001
        assert policy._preprocessor.call_args_list[1].kwargs == {}  # noqa: SLF001

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


class TestActionStatsSetup:
    """Training stats come from chunked data, not LeRobot frame metadata."""

    @pytest.mark.parametrize("mode", ["absolute", "delta"])
    def test_setup_collects_and_checkpoints_stats(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=4, action_mode=mode)
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {"state": Feature(name="state", ftype=FeatureType.STATE, shape=(1,))}
        dataset.action_features = {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))}
        dataset.stats = {
            ACTION: {"shape": (2,), "mean": [100.0, 100.0], "std": [1.0, 1.0]},
            "observation.state": {"shape": (1,), "mean": [0.0], "std": [1.0]},
        }
        actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        states = torch.tensor([[1.0], [3.0]])
        datamodule = types.SimpleNamespace(
            train_dataset=dataset, train_dataloader=lambda: [{ACTION: actions, STATE: states}]
        )
        policy._trainer = types.SimpleNamespace(datamodule=datamodule)  # noqa: SLF001
        sequence: list[str] = []

        def reformat(_policy: XR0, _data: object) -> None:
            sequence.append("chunked")

        def initialize() -> None:
            sequence.append("initialized")

        def rebuild(stats: dict[str, dict[str, Any]]) -> None:
            sequence.append("rebuilt")
            assert stats[ACTION]["mean"] == policy.hparams["dataset_stats"][ACTION]["mean"]

        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", reformat)
        monkeypatch.setattr(policy, "_initialize_model", initialize)
        monkeypatch.setattr(policy, "_rebuild_preprocessors", rebuild)
        policy.setup("fit")

        target = actions.clone()
        if mode == "delta":
            target[..., :1] -= states[:, None, :]
        assert sequence == ["initialized", "chunked", "rebuilt"]
        assert torch.allclose(torch.tensor(policy.hparams["dataset_stats"][ACTION]["mean"]), target.mean(0))
        assert torch.allclose(
            torch.tensor(policy.hparams["dataset_stats"][ACTION]["std"]), target.std(0, unbiased=False)
        )
        assert dataset.stats[ACTION]["mean"] == [100.0, 100.0]

    @pytest.mark.parametrize(
        ("shape", "missing", "message"),
        [
            (None, True, "requires a flat action feature"),
            (None, False, "requires a flat action feature"),
            ((2, 2), False, "requires a flat action feature"),
            ((4,), False, "exceeds max_action_dim"),
        ],
    )
    def test_setup_rejects_invalid_action_width(
        self, shape: tuple[int, ...] | None, missing: bool, message: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=3)
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {}
        dataset.action_features = (
            {} if missing else {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=shape)}
        )
        dataset.stats = {ACTION: {"shape": shape}}
        policy._trainer = types.SimpleNamespace(datamodule=types.SimpleNamespace(train_dataset=dataset))  # noqa: SLF001
        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", lambda *_args: None)
        scan = MagicMock()
        monkeypatch.setattr(xr0_policy, "compute_action_chunk_stats", scan)

        with pytest.raises(ValueError, match=message):
            policy.setup("fit")
        scan.assert_not_called()

    def test_setup_forwards_scan_options_and_slices_action_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        policy = XR0(
            chunk_size=2, n_action_steps=1, max_action_dim=3, action_mode="delta", action_stats_max_batches=4
        )
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {}
        dataset.action_features = {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))}
        dataset.stats = {ACTION: {"shape": (2,), "mean": [100.0, 100.0], "std": [1.0, 1.0]}}
        datamodule = types.SimpleNamespace(train_dataset=dataset)
        policy._trainer = types.SimpleNamespace(datamodule=datamodule)  # noqa: SLF001
        reformat = MagicMock()
        scan = MagicMock(
            return_value=(torch.tensor([[2, 3, 0], [4, 5, 0]]), torch.tensor([[1, 2, 1], [3, 4, 1]]))
        )
        initialize = MagicMock()
        rebuild = MagicMock()
        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", reformat)
        monkeypatch.setattr(xr0_policy, "compute_action_chunk_stats", scan)
        monkeypatch.setattr(policy, "_initialize_model", initialize)
        monkeypatch.setattr(policy, "_rebuild_preprocessors", rebuild)

        policy.setup("fit")

        reformat.assert_called_once_with(policy, datamodule)
        scan.assert_called_once_with(
            datamodule,
            chunk_size=2,
            action_dim=2,
            max_action_dim=3,
            action_mode="delta",
            max_batches=4,
            setup_stage=None,
            device="cpu",
        )
        stats = policy.hparams["dataset_stats"][ACTION]
        assert stats == {"shape": (2,), "mean": [[2, 3], [4, 5]], "std": [[1, 2], [3, 4]]}
        assert dataset.stats[ACTION]["mean"] == [100.0, 100.0]
        initialize.assert_called_once_with()
        rebuild.assert_called_once_with(policy.hparams["dataset_stats"])
        # Scanned stats are kept so a reloaded checkpoint skips the scan.
        assert policy.hparams["action_mean"] == [[2, 3], [4, 5]]
        assert policy.hparams["action_std"] == [[1, 2], [3, 4]]
        policy.setup("validate")
        scan.assert_called_once()

    def test_on_load_checkpoint_restores_action_stats(self) -> None:
        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=3)
        policy._action_mean = torch.zeros(2, 2)  # noqa: SLF001  # rescanned in setup
        policy._action_std = torch.ones(2, 2)  # noqa: SLF001
        policy.model = MagicMock()
        rebuild = MagicMock()
        policy._rebuild_preprocessors = rebuild  # noqa: SLF001
        saved = {ACTION: {"shape": (2,), "mean": [[2.0, 3.0], [4.0, 5.0]], "std": [[1.0, 2.0], [3.0, 4.0]]}}

        policy.on_load_checkpoint({"hyper_parameters": {"dataset_stats": saved}, "state_dict": {}})

        assert policy.hparams["action_mean"] == [[2.0, 3.0], [4.0, 5.0]]
        assert policy.hparams["action_std"] == [[1.0, 2.0], [3.0, 4.0]]
        assert policy.hparams["dataset_stats"] == saved
        rebuild.assert_called_once_with(saved)

    def test_setup_builds_action_stats_without_dataset_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An action feature with ``normalization_data=None`` is absent from Dataset.stats;
        # the scanned stats must still be stored, with metadata taken from the feature.
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=3)
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {}
        dataset.action_features = {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))}
        dataset.stats = {}
        policy._trainer = types.SimpleNamespace(datamodule=types.SimpleNamespace(train_dataset=dataset))  # noqa: SLF001
        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", lambda *_args: None)
        scan = MagicMock(return_value=(torch.tensor([[2, 3, 0], [4, 5, 0]]), torch.tensor([[1, 2, 1], [3, 4, 1]])))
        monkeypatch.setattr(xr0_policy, "compute_action_chunk_stats", scan)
        monkeypatch.setattr(policy, "_initialize_model", MagicMock())
        monkeypatch.setattr(policy, "_rebuild_preprocessors", MagicMock())

        policy.setup("fit")

        assert policy.hparams["dataset_stats"][ACTION] == {
            "type": FeatureType.ACTION.value,
            "name": ACTION,
            "shape": (2,),
            "mean": [[2, 3], [4, 5]],
            "std": [[1, 2], [3, 4]],
        }

    def test_explicit_stats_skip_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        mean = torch.zeros(2, 4)
        std = torch.ones(2, 4)
        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=4, action_mean=mean, action_std=std)
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {"state": Feature(name="state", ftype=FeatureType.STATE, shape=(3,))}
        dataset.action_features = {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))}
        dataset.stats = {ACTION: {"shape": (2,), "mean": [0.0, 0.0], "std": [1.0, 1.0]}}
        datamodule = types.SimpleNamespace(train_dataset=dataset, train_dataloader=MagicMock())
        policy._trainer = types.SimpleNamespace(datamodule=datamodule)  # noqa: SLF001
        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", lambda *_args: None)
        monkeypatch.setattr(policy, "_initialize_model", MagicMock())
        monkeypatch.setattr(policy, "_rebuild_preprocessors", MagicMock())

        policy.setup("fit")

        datamodule.train_dataloader.assert_not_called()
        # Padded ``max_action_dim``-wide overrides are trimmed to the real action width.
        assert policy.hparams["action_mean"] == mean[:, :2].tolist()
        assert policy.hparams["action_std"] == std[:, :2].tolist()
        assert policy.hparams["dataset_stats"][ACTION]["mean"] == mean[:, :2].tolist()

    @pytest.mark.parametrize("given", ["action_mean", "action_std"])
    def test_partial_explicit_stats_still_scan(self, given: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # Preprocessors ignore an incomplete override pair, so setup must scan.
        import physicalai.policies.xr0.policy as xr0_policy  # noqa: PLC0415

        policy = XR0(chunk_size=2, n_action_steps=1, max_action_dim=3, **{given: torch.ones(2, 3)})
        dataset = MagicMock(spec=Dataset)
        dataset.observation_features = {}
        dataset.action_features = {ACTION: Feature(name=ACTION, ftype=FeatureType.ACTION, shape=(2,))}
        dataset.stats = {ACTION: {"shape": (2,), "mean": [0.0, 0.0], "std": [1.0, 1.0]}}
        policy._trainer = types.SimpleNamespace(datamodule=types.SimpleNamespace(train_dataset=dataset))  # noqa: SLF001
        monkeypatch.setattr(xr0_policy, "reformat_dataset_to_match_policy", lambda *_args: None)
        scan = MagicMock(return_value=(torch.tensor([[2, 3, 0], [4, 5, 0]]), torch.tensor([[1, 2, 1], [3, 4, 1]])))
        monkeypatch.setattr(xr0_policy, "compute_action_chunk_stats", scan)
        monkeypatch.setattr(policy, "_initialize_model", MagicMock())
        monkeypatch.setattr(policy, "_rebuild_preprocessors", MagicMock())

        policy.setup("fit")

        scan.assert_called_once()
        assert policy.hparams["dataset_stats"][ACTION]["mean"] == [[2, 3], [4, 5]]
        assert policy.hparams["dataset_stats"][ACTION]["std"] == [[1, 2], [3, 4]]


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

    def test_openvino_export_preserves_image_view_map(self) -> None:
        view_map = {"images.top_camera": "ego", "images.wrist_camera": "wrist_left"}
        policy = XR0(image_key_view_map=view_map)
        policy.model = object()  # type: ignore[assignment]
        policy._preprocessor, policy._postprocessor = make_xr0_preprocessors()
        policy._preprocessor._processor = types.SimpleNamespace(  # noqa: SLF001
            image_processor=types.SimpleNamespace(
                image_mean=[0.0] * 3, image_std=[1.0] * 3,
                rescale_factor=1.0, patch_size=14, merge_size=2, temporal_patch_size=2,
            ),
        )

        spec = policy.extra_export_args["openvino"].preprocessors_specs[0]
        assert spec.model_dump()["image_key_view_map"] == view_map

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
        assert set(by_name) == {STATE, f"{IMAGES}.base", TASK}
        assert by_name[STATE].ftype is InferenceFeatureType.STATE
        assert by_name[STATE].shape == (8,)
        assert by_name[f"{IMAGES}.base"].ftype is InferenceFeatureType.VISUAL
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
        assert names == {STATE, f"{IMAGES}.base", f"{IMAGES}.wrist_left", TASK}

    def test_single_camera_sample_matches_view_map(self) -> None:
        # A single mapped camera must keep its name in the export sample, otherwise
        # the preprocessor's exact-key view-map check rejects it during export.
        from physicalai.policies.xr0.preprocessor import XR0Preprocessor  # noqa: PLC0415

        policy = XR0(chunk_size=30)
        policy.model = object()  # type: ignore[assignment]  # sentinel to bypass lazy-init guard
        policy._input_features, policy._output_features = XR0._stats_to_features(_minimal_export_stats())

        sample = policy.sample_input
        assert sample is not None
        views, _ = XR0Preprocessor(image_key_view_map={f"{IMAGES}.base": "ego"})._extract_view_images(sample)  # noqa: SLF001
        assert views == ["ego"]


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
        policy = XR0(action_mode="delta", action_mean=mean, action_std=std)
        if expect_stats:
            # Stored as float32 tensors for the preprocessor and mirrored into
            # hparams as plain lists so they round-trip through checkpoints.
            assert isinstance(policy._action_mean, torch.Tensor)
            assert isinstance(policy._action_std, torch.Tensor)
            assert policy._action_mean.dtype is torch.float32
            assert policy._action_std.dtype is torch.float32
            assert policy.hparams["action_mean"] == mean
            assert policy.hparams["action_std"] == std
        else:
            assert policy._action_mean is None
            assert policy._action_std is None
            # ``save_hyperparameters`` still captures the init args, but they are
            # left as ``None`` (not overwritten with the mirrored lists).
            assert policy.hparams["action_mean"] is None
            assert policy.hparams["action_std"] is None
