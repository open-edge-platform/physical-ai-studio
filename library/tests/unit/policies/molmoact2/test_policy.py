# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MolmoAct2 policy wrapper."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import lightning
import pytest
import torch

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.policies import get_policy
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config
from physicalai.policies.molmoact2.constants import (
    SO101_DEGREES_PER_NORMALIZED_UNIT,
    SO101_JOINT_OFFSETS,
    SO101_JOINT_SIGNS,
)


def test_registration_and_lazy_initialization() -> None:
    policy = get_policy("molmoact2")

    assert isinstance(policy, MolmoAct2)
    assert policy.model is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    assert policy.inputs_schema is None
    assert policy.outputs_schema is None


def test_private_processor_attributes_register_modules() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)
    preprocessor = torch.nn.Identity()
    postprocessor = torch.nn.Identity()

    policy._preprocessor = preprocessor  # type: ignore[assignment]
    policy._postprocessor = postprocessor  # type: ignore[assignment]

    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor
    assert "_preprocessor" in policy._modules
    assert "_postprocessor" in policy._modules


@pytest.mark.parametrize("method", ["forward", "predict_action_chunk", "compute_val_loss"])
def test_model_methods_require_initialization(method: str) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises((TypeError, RuntimeError), match="not initialized"):
        getattr(policy, method)(Observation(state=torch.zeros(1, 4)))


def test_invalid_lora_options() -> None:
    with pytest.raises(ValueError, match="requires use_lora"):
        MolmoAct2(pretrained_name_or_path=None, enable_lora_action_expert=True)
    with pytest.raises(ValueError, match="incompatible"):
        MolmoAct2(pretrained_name_or_path=None, use_lora=True, train_action_head_only=True)


@pytest.mark.parametrize(
    ("adapt_to_so101", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_so101_norm_tag_respects_explicit_adaptation_mode(
    adapt_to_so101: bool | None,
    expected: bool,
) -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=adapt_to_so101,
    )

    assert policy.adapt_to_so101 is expected


@pytest.mark.parametrize(
    ("norm_tag", "adapt_to_so101", "message"),
    [
        ("so100_so101_molmoact2", False, "requires adapt_to_so101=True"),
        ("other", True, "only supported with norm_tag"),
    ],
)
def test_pretrained_so101_stats_conversion_rejects_incompatible_modes(
    norm_tag: str,
    adapt_to_so101: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MolmoAct2(
            pretrained_name_or_path=None,
            norm_tag=norm_tag,
            adapt_to_so101=adapt_to_so101,
            convert_pretrained_so101_stats=True,
        )


def test_from_config_uses_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MolmoAct2Config(
        n_action_steps=3,
        chunk_size=5,
        use_random_input_noise=True,
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=True,
        convert_pretrained_so101_stats=True,
    )
    initialized: list[MolmoAct2Config] = []

    def initialize(policy: MolmoAct2, policy_config: MolmoAct2Config) -> None:
        policy.config = policy_config
        initialized.append(policy_config)

    monkeypatch.setattr(MolmoAct2, "_initialize_from_config", initialize)

    policy = MolmoAct2.from_config(
        config,
        preserve_pretrained_normalization_in_training=True,
        compile_model=True,
        optimizer_lr=2e-5,
    )

    assert initialized == [config]
    assert policy.pretrained_name_or_path is None
    assert (policy.n_action_steps, policy.chunk_size) == (3, 5)
    assert policy.preserve_pretrained_normalization_in_training is True
    assert policy.convert_pretrained_so101_stats is True
    assert policy.compile_model is True
    assert policy.optimizer_lr == 2e-5


def test_explicit_features_override_norm_tag_features_without_inheriting_statistics(tmp_path: Path) -> None:
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 600, 800)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(4,)),
    ]
    output_features = [Feature(name="action", ftype=FeatureType.ACTION, shape=(4,))]
    policy = MolmoAct2(pretrained_name_or_path=None, norm_tag="test")
    policy.input_features = input_features
    policy.output_features = output_features
    norm_stats = {
        "metadata_by_tag": {
            "test": {
                "camera_keys": [],
                "state_key": "observation.state",
                "state_stats": {"q01": [-1.0] * 4, "q99": [1.0] * 4},
                "action_key": "action",
                "action_stats": {"q01": [-1.0] * 4, "q99": [1.0] * 4},
                "action_horizon": 30,
                "normalize_gripper": True,
            },
        },
    }

    config = policy._convert_config({}, norm_stats, {}, tmp_path)

    assert config.input_features is not None
    assert config.output_features is not None
    assert config.input_features[0] == input_features[0]
    assert config.input_features[1].name == "state"
    assert config.input_features[1].shape == (4,)
    assert config.input_features[1].normalization_data is None
    assert config.output_features[0].name == "action"
    assert config.output_features[0].shape == (4,)
    assert config.output_features[0].normalization_data is None


def test_convert_config_corrects_pretrained_so101_statistics_once(tmp_path: Path) -> None:
    checkpoint_q01 = [-40.0, 50.0, 40.0, -30.0, -20.0, 2.0]
    checkpoint_q99 = [45.0, 180.0, 170.0, 35.0, 30.0, 95.0]
    norm_stats = {
        "metadata_by_tag": {
            "so100_so101_molmoact2": {
                "camera_keys": [],
                "state_key": "observation.state",
                "state_stats": {"q01": checkpoint_q01, "q99": checkpoint_q99},
                "action_key": "action",
                "action_stats": {"q01": checkpoint_q01, "q99": checkpoint_q99},
                "action_horizon": 30,
                "normalize_gripper": True,
            },
        },
    }
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=True,
        convert_pretrained_so101_stats=True,
    )

    config = policy._convert_config({}, norm_stats, {}, tmp_path)

    assert config.convert_pretrained_so101_stats is True
    assert config.input_features is not None
    assert config.output_features is not None
    state_stats = config.input_features[-1].normalization_data
    action_stats = config.output_features[0].normalization_data
    assert state_stats is not None
    assert action_stats is not None
    offsets = [0.0, 90.0, 90.0, 0.0, 0.0]
    expected_q01 = [
        offset + (value - offset) / scale
        for value, offset, scale in zip(
            checkpoint_q01[:5],
            offsets,
            SO101_DEGREES_PER_NORMALIZED_UNIT,
            strict=True,
        )
    ] + [checkpoint_q01[-1]]
    expected_q99 = [
        offset + (value - offset) / scale
        for value, offset, scale in zip(
            checkpoint_q99[:5],
            offsets,
            SO101_DEGREES_PER_NORMALIZED_UNIT,
            strict=True,
        )
    ] + [checkpoint_q99[-1]]
    assert state_stats.q01 == pytest.approx(expected_q01)
    assert state_stats.q99 == pytest.approx(expected_q99)
    assert action_stats == state_stats

    restored = MolmoAct2Config.from_dict(config.to_dict())

    assert restored.input_features[-1].normalization_data == state_stats
    assert restored.output_features[0].normalization_data == action_stats


def test_set_features_copies_only_requested_state_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    state_feature = tiny_molmoact2_config.input_features[-1]
    config_without_images = replace(tiny_molmoact2_config, input_features=[state_feature])
    policy = MolmoAct2.from_config(config_without_images).eval()
    model = policy.model
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    replacement_state_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    replacement_action_stats = NormalizationParameters(q01=[-3.0] * 4, q99=[3.0] * 4)
    input_features = [
        Feature(name="overview", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="left_wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="right_wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(
            name="robot_state",
            ftype=FeatureType.STATE,
            shape=(4,),
            normalization_data=replacement_state_stats,
        ),
    ]
    output_features = [
        Feature(
            name="robot_action",
            ftype=FeatureType.ACTION,
            shape=(4,),
            normalization_data=replacement_action_stats,
        ),
    ]

    policy.set_features(
        input_features,
        output_features,
        copy_state_normalization=True,
    )

    assert policy.model is model
    assert policy._preprocessor is not preprocessor
    assert policy._postprocessor is not postprocessor
    assert policy._preprocessor is not None and not policy._preprocessor.training
    assert policy._postprocessor is not None and not policy._postprocessor.training
    assert policy.config is not None
    assert policy.config.input_features == policy.input_features
    assert policy.config.output_features == policy.output_features
    assert [feature.name for feature in policy.input_features or []] == [
        "overview",
        "left_wrist",
        "right_wrist",
        "robot_state",
    ]
    assert policy.input_features is not None
    assert policy.output_features is not None
    assert policy.input_features[-1].normalization_data == state_feature.normalization_data
    assert policy.output_features[0].normalization_data is replacement_action_stats


def test_set_features_ignores_training_normalization_preservation(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(
        tiny_molmoact2_config,
        preserve_pretrained_normalization_in_training=True,
    )
    replacement_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    input_features = [
        replace(tiny_molmoact2_config.input_features[0], name="dataset_camera"),
        replace(tiny_molmoact2_config.input_features[-1], normalization_data=replacement_stats),
    ]
    output_features = [
        replace(tiny_molmoact2_config.output_features[0], normalization_data=replacement_stats),
    ]

    policy.set_features(input_features, output_features)

    assert policy.input_features[-1].normalization_data is replacement_stats
    assert policy.output_features[0].normalization_data is replacement_stats


def test_set_features_copies_only_requested_action_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    replacement_state_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    replacement_action_stats = NormalizationParameters(q01=[-3.0] * 4, q99=[3.0] * 4)
    input_features = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(
            name="robot_state",
            ftype=FeatureType.STATE,
            shape=(4,),
            normalization_data=replacement_state_stats,
        ),
    ]
    output_features = [
        Feature(
            name="robot_action",
            ftype=FeatureType.ACTION,
            shape=(4,),
            normalization_data=replacement_action_stats,
        ),
    ]

    policy.set_features(
        input_features,
        output_features,
        copy_action_normalization=True,
    )

    assert policy.input_features is not None
    assert policy.output_features is not None
    assert policy.input_features[-1].normalization_data is replacement_state_stats
    assert policy.output_features[0].normalization_data == tiny_molmoact2_config.output_features[0].normalization_data


def test_set_features_transforms_dataset_normalization_in_adapted_mode(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=True)
    policy = MolmoAct2.from_config(config)
    state_stats = NormalizationParameters(
        q01=[-2.0, -3.0, -4.0, -5.0],
        q99=[2.0, 3.0, 4.0, 5.0],
    )
    action_stats = NormalizationParameters(
        q01=[-12.0, -13.0, -14.0, -15.0],
        q99=[12.0, 13.0, 14.0, 15.0],
    )
    input_features = [
        replace(feature, normalization_data=state_stats) if feature.ftype == FeatureType.STATE else feature
        for feature in config.input_features
    ]
    output_features = [replace(config.output_features[0], normalization_data=action_stats)]

    policy.set_features(input_features, output_features)

    resolved_state = policy.input_features[-1].normalization_data
    resolved_action = policy.output_features[0].normalization_data
    assert resolved_state is not None
    assert resolved_action is not None
    assert resolved_state.q01 == [-2.0, 87.0, 86.0, -5.0]
    assert resolved_state.q99 == [2.0, 93.0, 94.0, 5.0]
    assert resolved_action.q01 == [-12.0, 77.0, 76.0, -15.0]
    assert resolved_action.q99 == [12.0, 103.0, 104.0, 15.0]
    assert input_features[-1].normalization_data is state_stats
    assert output_features[0].normalization_data is action_stats


def test_set_features_does_not_transform_copied_policy_normalization_twice(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=True)
    policy = MolmoAct2.from_config(config)
    replacement_inputs = [
        replace(feature, normalization_data=None) if feature.ftype == FeatureType.STATE else feature
        for feature in config.input_features
    ]

    policy.set_features(
        replacement_inputs,
        list(config.output_features),
        copy_state_normalization=True,
        copy_action_normalization=True,
    )

    assert policy.input_features[-1].normalization_data == config.input_features[-1].normalization_data
    assert policy.output_features[0].normalization_data == config.output_features[0].normalization_data


def test_set_features_rejects_incompatible_normalization_shape_atomically(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    config = policy.config
    input_features = policy.input_features
    preprocessor = policy._preprocessor
    replacement_inputs = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(5,)),
    ]

    with pytest.raises(ValueError, match="Cannot copy STATE normalization"):
        policy.set_features(
            replacement_inputs,
            list(policy.output_features or []),
            copy_state_normalization=True,
        )

    assert policy.config is config
    assert policy.input_features is input_features
    assert policy._preprocessor is preprocessor


def test_set_features_requires_initialized_policy() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises(TypeError, match="not initialized"):
        policy.set_features([], [])


def test_rename_features_matches_libero_camera_names(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    wrist_feature = replace(tiny_molmoact2_config.input_features[0], name="wrist_image")
    config = replace(
        tiny_molmoact2_config,
        input_features=[
            tiny_molmoact2_config.input_features[0],
            wrist_feature,
            tiny_molmoact2_config.input_features[-1],
        ],
    )
    policy = MolmoAct2.from_config(config).eval()
    model = policy.model
    output_features = policy.output_features
    state_normalization = config.input_features[-1].normalization_data

    policy.rename_features({"wrist_image": "image2"})

    assert policy.model is model
    assert policy.output_features == output_features
    assert not policy.training
    assert policy.config is not None
    assert [feature.name for feature in policy.input_features or []] == ["image", "image2", "state"]
    assert policy.input_features is not None
    assert policy.input_features[1] == replace(wrist_feature, name="image2")
    assert policy.input_features[-1].normalization_data is state_normalization
    assert policy.config.input_features == policy.input_features
    assert policy._preprocessor is not None
    assert policy._preprocessor._extractor.image_keys == ["image", "image2"]

    observation = Observation(
        images={
            "image": torch.zeros(1, 3, 28, 28),
            "image2": torch.ones(1, 3, 28, 28),
        },
        state=torch.zeros(1, 4),
        task=["pick up the object"],
    )
    extracted = policy._preprocessor._extractor.extract(observation.to_dict())
    torch.testing.assert_close(extracted.images_by_example[0][0], torch.zeros(3, 28, 28))
    torch.testing.assert_close(extracted.images_by_example[0][1], torch.ones(3, 28, 28))


def test_rename_features_supports_swaps_and_empty_mapping(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    second_camera = replace(tiny_molmoact2_config.input_features[0], name="wrist_image")
    config = replace(
        tiny_molmoact2_config,
        input_features=[
            tiny_molmoact2_config.input_features[0],
            second_camera,
            tiny_molmoact2_config.input_features[-1],
        ],
    )
    policy = MolmoAct2.from_config(config)

    policy.rename_features({})
    preprocessor = policy._preprocessor
    policy.rename_features({"image": "wrist_image", "wrist_image": "image"})

    assert preprocessor is not None
    assert policy._preprocessor is not preprocessor
    assert [feature.name for feature in policy.input_features or []] == ["wrist_image", "image", "state"]


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"missing": "camera"}, "unknown input features"),
        ({"image": ""}, "Replacement feature names"),
        ({"image": 1}, "Replacement feature names"),
        ({"image": "state"}, "duplicate input names"),
    ],
)
def test_rename_features_rejects_invalid_mapping_atomically(
    tiny_molmoact2_config: MolmoAct2Config,
    mapping: dict[str, str],
    message: str,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    config = policy.config
    input_features = policy.input_features
    preprocessor = policy._preprocessor

    with pytest.raises(ValueError, match=message):
        policy.rename_features(mapping)

    assert policy.config is config
    assert policy.input_features is input_features
    assert policy._preprocessor is preprocessor


def test_rename_features_rolls_back_when_processor_creation_fails(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    config = policy.config
    input_features = policy.input_features
    preprocessor = policy._preprocessor

    def fail_to_create_processors(_config: MolmoAct2Config) -> None:
        msg = "processor creation failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "physicalai.policies.molmoact2.policy.make_molmoact2_preprocessors",
        fail_to_create_processors,
    )

    with pytest.raises(RuntimeError, match="processor creation failed"):
        policy.rename_features({"image": "camera"})

    assert policy.config is config
    assert policy.input_features is input_features
    assert policy._preprocessor is preprocessor


def test_rename_features_requires_initialized_policy() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises(TypeError, match="not initialized"):
        policy.rename_features({"image": "camera"})


def test_setup_replaces_eager_normalization_with_dataset_normalization(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    model = policy.model
    dataset_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    dataset_inputs = [
        replace(feature, normalization_data=dataset_stats)
        for feature in tiny_molmoact2_config.input_features
    ]
    dataset_outputs = [
        replace(feature, normalization_data=dataset_stats)
        for feature in tiny_molmoact2_config.output_features
    ]
    train_dataset = Mock(spec=Dataset)
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))

    with caplog.at_level("WARNING"):
        policy.setup("fit")

    assert "replacing them with the dataset features" in caplog.text
    assert policy.model is model
    assert policy.input_features == dataset_inputs
    assert policy.output_features == dataset_outputs
    assert policy.config is not None
    assert policy.config.input_features == dataset_inputs
    assert policy.config.output_features == dataset_outputs


def test_setup_preserves_pretrained_normalization_with_dataset_feature_contract(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = MolmoAct2.from_config(
        tiny_molmoact2_config,
        preserve_pretrained_normalization_in_training=True,
    )
    pretrained_state_stats = tiny_molmoact2_config.input_features[-1].normalization_data
    pretrained_action_stats = tiny_molmoact2_config.output_features[0].normalization_data
    dataset_stats = NormalizationParameters(q01=[-2.0] * 4, q99=[2.0] * 4)
    visual_feature = tiny_molmoact2_config.input_features[0]
    dataset_inputs = [
        replace(visual_feature, name="wrist"),
        replace(visual_feature, name="overview"),
        replace(tiny_molmoact2_config.input_features[-1], normalization_data=dataset_stats),
    ]
    dataset_outputs = [replace(tiny_molmoact2_config.output_features[0], normalization_data=dataset_stats)]
    trainer = Mock()
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))

    policy.setup("fit")

    expected_input_names = ["wrist", "overview", tiny_molmoact2_config.input_features[-1].name]
    assert [feature.name for feature in policy.input_features] == expected_input_names
    assert policy.config is not None
    assert [feature.name for feature in policy.config.input_features] == expected_input_names
    assert policy._preprocessor is not None
    assert policy._preprocessor._extractor.image_keys == ["wrist", "overview"]
    assert policy.input_features[-1].normalization_data == pretrained_state_stats
    assert policy.output_features[0].normalization_data == pretrained_action_stats


def test_setup_preserves_checkpoint_frame_normalization_without_transforming_twice(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=True)
    policy = MolmoAct2.from_config(config, preserve_pretrained_normalization_in_training=True)
    dataset_stats = NormalizationParameters(
        q01=[-2.0, -3.0, -4.0, -5.0],
        q99=[2.0, 3.0, 4.0, 5.0],
    )
    dataset_inputs = [
        replace(feature, normalization_data=dataset_stats)
        if feature.ftype == FeatureType.STATE
        else feature
        for feature in config.input_features
    ]
    dataset_outputs = [replace(config.output_features[0], normalization_data=dataset_stats)]
    trainer = Mock()
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))

    policy.setup("fit")

    assert policy.input_features[-1].normalization_data == config.input_features[-1].normalization_data
    assert policy.output_features[0].normalization_data == config.output_features[0].normalization_data


def test_setup_uses_dataset_normalization_when_uninitialized(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        preserve_pretrained_normalization_in_training=True,
    )
    dataset_inputs = list(tiny_molmoact2_config.input_features)
    dataset_outputs = list(tiny_molmoact2_config.output_features)
    trainer = Mock()
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    initialize_model = Mock()
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))
    monkeypatch.setattr(policy, "initialize_model", initialize_model)

    policy.setup("fit")

    assert "missing q01/q99 statistics" not in caplog.text
    assert policy.input_features == dataset_inputs
    assert policy.output_features == dataset_outputs
    initialize_model.assert_called_once_with()


def test_setup_warns_when_dataset_quantiles_are_missing(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)
    dataset_inputs = list(tiny_molmoact2_config.input_features)
    dataset_outputs = [
        replace(
            tiny_molmoact2_config.output_features[0],
            normalization_data=NormalizationParameters(mean=[0.0] * 4, std=[1.0] * 4),
        ),
    ]
    trainer = Mock()
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))
    monkeypatch.setattr(policy, "initialize_model", Mock())

    with caplog.at_level("WARNING"):
        policy.setup("fit")

    assert "missing q01/q99 statistics for: action" in caplog.text
    assert "lerobot.scripts.augment_dataset_quantile_stats" in caplog.text


def test_setup_transforms_dataset_normalization_in_adapted_mode(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=True)
    policy = MolmoAct2.from_config(config)
    dataset_stats = NormalizationParameters(
        q01=[-2.0, -3.0, -4.0, -5.0],
        q99=[2.0, 3.0, 4.0, 5.0],
    )
    dataset_inputs = [
        replace(feature, normalization_data=dataset_stats)
        if feature.ftype == FeatureType.STATE
        else feature
        for feature in config.input_features
    ]
    dataset_outputs = [replace(config.output_features[0], normalization_data=dataset_stats)]
    trainer = Mock()
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _dataset: (dataset_inputs, dataset_outputs))

    policy.setup("fit")

    state_stats = policy.input_features[-1].normalization_data
    action_stats = policy.output_features[0].normalization_data
    assert state_stats is not None and state_stats.q01 == [-2.0, 87.0, 86.0, -5.0]
    assert action_stats is not None and action_stats.q99 == [2.0, 93.0, 94.0, 5.0]


def test_setup_replaces_eager_feature_contract_with_dataset_contract(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    model = policy.model
    dataset_inputs = [
        replace(tiny_molmoact2_config.input_features[0], name="other_camera"),
        *tiny_molmoact2_config.input_features[1:],
    ]
    train_dataset = Mock(spec=Dataset)
    trainer = Mock()
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer
    monkeypatch.setattr(
        policy,
        "_dataset_features",
        lambda _dataset: (dataset_inputs, tiny_molmoact2_config.output_features),
    )

    with caplog.at_level("WARNING"):
        policy.setup("fit")

    assert "replacing them with the dataset features" in caplog.text
    assert policy.model is model
    assert policy.input_features == dataset_inputs
    assert policy.output_features == tiny_molmoact2_config.output_features


def test_load_from_checkpoint_restores_config_and_weights(
    tiny_molmoact2_config: MolmoAct2Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = MolmoAct2.from_config(
        tiny_molmoact2_config,
        preserve_pretrained_normalization_in_training=True,
    )
    checkpoint = {
        "state_dict": policy.state_dict(),
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": dict(policy.hparams),
    }
    policy.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "molmoact2.ckpt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only trusted data.
    torch.save(checkpoint, checkpoint_path)

    def fail_pretrained_resolution(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Lightning checkpoint loading must not resolve pretrained assets")

    monkeypatch.setattr(MolmoAct2, "_from_hf", fail_pretrained_resolution)

    restored = MolmoAct2.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert restored.config == tiny_molmoact2_config
    assert restored.preserve_pretrained_normalization_in_training is True
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_load_from_checkpoint_preserves_normalization_and_training_arguments(
    tiny_molmoact2_config: MolmoAct2Config,
    tmp_path: Path,
) -> None:
    adapted_config = replace(
        tiny_molmoact2_config,
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=True,
        convert_pretrained_so101_stats=True,
    )
    policy = MolmoAct2.from_config(
        adapted_config,
        preserve_pretrained_normalization_in_training=True,
        optimizer_lr=1e-5,
        optimizer_vit_lr=5e-6,
        optimizer_connector_lr=5e-6,
        optimizer_action_expert_lr=5e-5,
        scheduler_warmup_steps=200,
        scheduler_decay_steps=30_000,
        scheduler_decay_lr=1e-6,
    )
    checkpoint = {
        "state_dict": policy.state_dict(),
        "pytorch-lightning_version": lightning.__version__,
        "hyper_parameters": dict(policy.hparams),
    }
    policy.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "molmoact2-training-arguments.ckpt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only trusted data.
    torch.save(checkpoint, checkpoint_path)

    restored = MolmoAct2.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert restored.preserve_pretrained_normalization_in_training is True
    assert restored.adapt_to_so101 is True
    assert restored.convert_pretrained_so101_stats is True
    assert restored.config is not None and restored.config.adapt_to_so101 is True
    assert restored.config.convert_pretrained_so101_stats is True
    assert restored.input_features[-1].normalization_data == policy.input_features[-1].normalization_data
    assert restored.output_features[0].normalization_data == policy.output_features[0].normalization_data
    assert restored.n_action_steps == policy.n_action_steps
    assert restored.chunk_size == policy.chunk_size
    assert restored.optimizer_lr == 1e-5
    assert restored.optimizer_vit_lr == 5e-6
    assert restored.optimizer_connector_lr == 5e-6
    assert restored.optimizer_action_expert_lr == 5e-5
    assert restored.scheduler_warmup_steps == 200
    assert restored.scheduler_decay_steps == 30_000
    assert restored.scheduler_decay_lr == 1e-6


def test_configure_optimizers_uses_estimated_step_budget(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(
        tiny_molmoact2_config,
        optimizer_lr=1e-5,
        optimizer_vit_lr=2e-5,
        optimizer_connector_lr=3e-5,
        optimizer_action_expert_lr=5e-5,
        scheduler_warmup_steps=200,
        scheduler_decay_steps=30_000,
        scheduler_decay_lr=1e-6,
    )
    trainer = Mock()
    trainer.estimated_stepping_batches = 3_000
    policy._trainer = trainer

    configured = policy.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler_config = configured["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]

    assert scheduler_config["interval"] == "step"
    assert all(group["lr"] == pytest.approx(group["initial_lr"] / 20) for group in optimizer.param_groups)
    assert scheduler.get_last_lr() == pytest.approx([group["lr"] for group in optimizer.param_groups])
    assert scheduler.lr_lambdas[0](2_999) == pytest.approx(1e-6 / 1e-5)


def test_optimizer_defaults_match_so101_finetuning_recipe() -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    assert policy.optimizer_lr == 5e-5
    assert policy.optimizer_vit_lr == 5e-5
    assert policy.optimizer_connector_lr == 5e-5
    assert policy.optimizer_action_expert_lr == 5e-5
    assert policy.scheduler_warmup_steps == 200
    assert policy.scheduler_decay_steps == 30_000


@pytest.mark.parametrize("policy_config", [None, "invalid"])
def test_load_checkpoint_requires_policy_config(policy_config: object) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None)

    with pytest.raises(TypeError, match="valid policy_config"):
        policy.on_load_checkpoint({"policy_config": policy_config})


def test_restore_checkpoint_rejects_different_initialized_config(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    different_config = replace(tiny_molmoact2_config, n_action_steps=1)

    with pytest.raises(ValueError, match="does not match"):
        policy._restore_policy_config(different_config.to_dict())


def test_runtime_options_are_policy_owned() -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        compile_model=True,
        openvino_compress_to_fp16=True,
        gradient_checkpointing=True,
        optimizer_lr=2e-5,
    )

    assert policy.compile_model is True
    assert policy.openvino_compress_to_fp16 is True
    assert policy.gradient_checkpointing is True
    assert policy.optimizer_lr == 2e-5
    assert "compile_model" not in policy.hparams


def test_sample_input_zeros_only_state_with_passthrough_dimensions(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic_sample = {
        "image": torch.full((1, 3, 28, 28), 0.5),
        "state": torch.full((1, 4), 2.0),
        "task": "Example prompt string",
    }
    monkeypatch.setattr(ExportablePolicyMixin, "sample_input", property(lambda _self: dict(generic_sample)))
    state_feature = tiny_molmoact2_config.input_features[-1]
    masked_state = replace(
        state_feature,
        normalization_data=replace(
            state_feature.normalization_data,
            mask=[True, True, True, False],
        ),
    )
    masked_config = replace(
        tiny_molmoact2_config,
        input_features=[*tiny_molmoact2_config.input_features[:-1], masked_state],
    )

    masked_sample = MolmoAct2.from_config(masked_config).sample_input
    normalized_sample = MolmoAct2.from_config(tiny_molmoact2_config).sample_input

    assert masked_sample is not None
    assert normalized_sample is not None
    torch.testing.assert_close(masked_sample["state"], torch.zeros(1, 4))
    torch.testing.assert_close(normalized_sample["state"], generic_sample["state"])
    torch.testing.assert_close(masked_sample["image"], generic_sample["image"])
    assert masked_sample["task"] == generic_sample["task"]


@pytest.mark.parametrize("trim_actions", [True, False])
def test_export_uses_action_chunk_trimmer_for_shorter_execution_horizon(
    tiny_molmoact2_config: MolmoAct2Config,
    trim_actions: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_action_steps = tiny_molmoact2_config.n_action_steps if trim_actions else tiny_molmoact2_config.chunk_size
    config = replace(tiny_molmoact2_config, n_action_steps=n_action_steps)
    policy = MolmoAct2.from_config(config)
    monkeypatch.setattr(policy, "_openvino_token_ids", lambda: (1, 0, [10, 11, 12]))

    assert policy.outputs_schema is not None
    assert policy.outputs_schema[0].shape == (config.chunk_size, *config.output_features[0].shape)

    export_args = policy.extra_export_args
    torch_postprocessors = export_args[ExportBackend.TORCH].postprocessors_specs
    openvino_postprocessors = export_args[ExportBackend.OPENVINO].postprocessors_specs
    assert [spec.type for spec in torch_postprocessors] == (["action_chunk_trimmer"] if trim_actions else [])
    assert [spec.type for spec in openvino_postprocessors] == [
        "molmoact2_postprocess",
        *(["action_chunk_trimmer"] if trim_actions else []),
    ]
    if trim_actions:
        assert torch_postprocessors[0].n_action_steps == config.n_action_steps
        assert openvino_postprocessors[-1].n_action_steps == config.n_action_steps


def test_openvino_compression_is_used_by_export(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2(pretrained_name_or_path=None, openvino_compress_to_fp16=True)
    policy.config = tiny_molmoact2_config
    policy.input_features = tiny_molmoact2_config.input_features
    policy.output_features = tiny_molmoact2_config.output_features
    policy.model = Mock()
    policy._preprocessor = Mock()
    policy._preprocessor.tokenizer.bos_token_id = 1
    policy._preprocessor.tokenizer.pad_token_id = 0

    export_args = policy.extra_export_args[ExportBackend.OPENVINO]

    assert export_args.compress_to_fp16 is True


def test_openvino_export_forwards_runtime_input_config(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    config = replace(
        tiny_molmoact2_config,
        frame_start_token_id=21,
        frame_end_token_id=22,
        image_low_res_id=23,
    )
    policy = MolmoAct2(pretrained_name_or_path=None)
    policy.config = config
    policy.input_features = config.input_features
    policy.output_features = config.output_features
    policy.model = Mock()
    policy._preprocessor = Mock()
    policy._preprocessor.tokenizer.bos_token_id = 1
    policy._preprocessor.tokenizer.pad_token_id = 7

    export_args = policy.extra_export_args[ExportBackend.OPENVINO]
    model_inputs = next(spec for spec in export_args.preprocessors_specs if spec.type == "molmoact2_inputs")

    assert model_inputs.pad_token_id == 7
    assert model_inputs.frame_start_token_id == 21
    assert model_inputs.frame_end_token_id == 22
    assert model_inputs.image_low_res_id == 23


@pytest.mark.parametrize("adapt_to_so101", [True, False])
def test_openvino_export_preserves_resolved_so101_mode_and_statistics(
    tiny_molmoact2_config: MolmoAct2Config,
    adapt_to_so101: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(tiny_molmoact2_config, adapt_to_so101=adapt_to_so101)
    policy = MolmoAct2.from_config(config)
    raw_stats = NormalizationParameters(
        q01=[-2.0, -3.0, -4.0, -5.0],
        q99=[2.0, 3.0, 4.0, 5.0],
    )
    input_features = [
        replace(feature, normalization_data=raw_stats) if feature.ftype == FeatureType.STATE else feature
        for feature in config.input_features
    ]
    output_features = [replace(config.output_features[0], normalization_data=raw_stats)]
    policy.set_features(input_features, output_features)
    monkeypatch.setattr(policy, "_openvino_token_ids", lambda: (1, 0, [10, 11, 12]))

    export_args = policy.extra_export_args[ExportBackend.OPENVINO]
    preprocessor = next(spec for spec in export_args.preprocessors_specs if spec.type == "molmoact2")
    postprocessor = next(spec for spec in export_args.postprocessors_specs if spec.type == "molmoact2_postprocess")
    preprocessor_types = [spec.type for spec in export_args.preprocessors_specs]
    postprocessor_types = [spec.type for spec in export_args.postprocessors_specs]
    expected_q01 = [-2.0, 87.0, 86.0, -5.0] if adapt_to_so101 else raw_stats.q01
    expected_q99 = [2.0, 93.0, 94.0, 5.0] if adapt_to_so101 else raw_stats.q99

    assert (preprocessor_types[0] == "joint_frame_preprocess") is adapt_to_so101
    assert (postprocessor_types[:2] == ["molmoact2_postprocess", "joint_frame_postprocess"]) is adapt_to_so101
    if adapt_to_so101:
        joint_preprocessor = export_args.preprocessors_specs[0]
        joint_postprocessor = export_args.postprocessors_specs[1]
        assert joint_preprocessor.feature == "state"
        assert joint_postprocessor.feature == "action"
        assert joint_preprocessor.signs == list(SO101_JOINT_SIGNS)
        assert joint_postprocessor.signs == list(SO101_JOINT_SIGNS)
        assert joint_preprocessor.offsets == list(SO101_JOINT_OFFSETS)
        assert joint_postprocessor.offsets == list(SO101_JOINT_OFFSETS)
    assert preprocessor.state_stats["q01"] == expected_q01
    assert preprocessor.state_stats["q99"] == expected_q99
    assert postprocessor.action_stats["q01"] == expected_q01
    assert postprocessor.action_stats["q99"] == expected_q99


def test_openvino_export_uses_corrected_pretrained_so101_statistics(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        tiny_molmoact2_config,
        norm_tag="so100_so101_molmoact2",
        adapt_to_so101=True,
        convert_pretrained_so101_stats=True,
    )
    policy = MolmoAct2.from_config(config)
    monkeypatch.setattr(policy, "_openvino_token_ids", lambda: (1, 0, [10, 11, 12]))

    export_args = policy.extra_export_args[ExportBackend.OPENVINO]
    preprocessor = next(spec for spec in export_args.preprocessors_specs if spec.type == "molmoact2")
    postprocessor = next(spec for spec in export_args.postprocessors_specs if spec.type == "molmoact2_postprocess")
    state_stats = config.input_features[-1].normalization_data
    action_stats = config.output_features[0].normalization_data
    assert state_stats is not None
    assert action_stats is not None

    assert export_args.preprocessors_specs[0].type == "joint_frame_preprocess"
    assert export_args.postprocessors_specs[:2] == [
        postprocessor,
        next(spec for spec in export_args.postprocessors_specs if spec.type == "joint_frame_postprocess"),
    ]
    assert preprocessor.state_stats == {
        "q01": state_stats.q01,
        "q99": state_stats.q99,
    }
    assert postprocessor.action_stats == {
        "q01": action_stats.q01,
        "q99": action_stats.q99,
    }


def test_model_modifications_are_applied_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = MolmoAct2(
        pretrained_name_or_path=None,
        compile_model=True,
        gradient_checkpointing=True,
        use_lora=True,
        train_action_head_only=False,
    )
    model = Mock()
    monkeypatch.setattr(policy, "_require_model", lambda: model)

    policy._apply_model_modifications()

    model.enable_gradient_checkpointing.assert_called_once_with()
    model.enable_lora.assert_called_once_with(enable_action_expert=False)
    model.enable_compile.assert_called_once_with()


def test_supported_export_backends() -> None:
    assert MolmoAct2.get_supported_export_backends() == [ExportBackend.TORCH, ExportBackend.OPENVINO]
