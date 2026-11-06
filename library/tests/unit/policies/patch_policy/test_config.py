# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Patch Policy configuration contract."""

from dataclasses import replace
from pathlib import Path

import pytest

from physicalai.config import Config
from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.patch_policy.config import PatchPolicyConfig, resolve_action_dim


def _config() -> PatchPolicyConfig:
    return PatchPolicyConfig(
        input_features=[Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, 8, 8))],
        output_features=[
            Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(2,),
                normalization_data=NormalizationParameters(
                    mean=[1.0, 2.0],
                    std=[2.0, 3.0],
                    min=[-1.0, -2.0],
                    max=[3.0, 4.0],
                    q01=[-0.5, -1.5],
                    q99=[2.5, 3.5],
                    mask=[True, False],
                ),
            ),
        ],
        action_dim=2,
    )


def test_dictionary_roundtrip_preserves_feature_types_shapes_and_statistics() -> None:
    config = _config()

    assert isinstance(config, Config)
    wire = config.to_dict()
    assert wire["output_features"][0]["ftype"] == "ACTION"
    restored = PatchPolicyConfig.from_dict(wire)

    assert restored == config
    assert isinstance(restored.output_features[0], Feature)
    assert restored.output_features[0].ftype is FeatureType.ACTION
    assert restored.output_features[0].shape == (2,)
    assert isinstance(restored.output_features[0].normalization_data, NormalizationParameters)


@pytest.mark.parametrize("format", ["dict", "jsonargparse"])
def test_yaml_roundtrip_preserves_architecture_and_sampling_options(tmp_path: Path, format: str) -> None:
    config = replace(
        _config(),
        encoder_name="webssl",
        encoder_output="average",
        encoder_config={"architecture": "tiny", "depth": 2},
        image_size=(8, 12),
        action_head="diffusion",
        n_obs_steps=2,
        chunk_size=3,
        n_action_steps=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0.1,
        embedding_dropout=0.2,
        num_train_timesteps=8,
        num_inference_steps=4,
        beta_schedule="linear",
        prediction_type="sample",
        clip_sample=False,
        clip_sample_range=2.0,
        eta=0.0,
        use_random_input_noise=False,
        vq_latent_dim=4,
        vq_num_embeddings=3,
        vq_num_groups=1,
        vq_action_scale=2.0,
        offset_loss_weight=3.0,
        secondary_code_weight=0.25,
        focal_loss_gamma=1.0,
    )
    path = tmp_path / "patch.yaml"
    config.save(path, format=format)

    assert PatchPolicyConfig.load(path) == config
    assert PatchPolicyConfig.load(config.to_jsonargparse()) == config


def test_action_dimension_flattens_features_including_matrix_shapes() -> None:
    features = [
        Feature(name="arm", ftype=FeatureType.ACTION, shape=(2, 3)),
        Feature(name="gripper", ftype=FeatureType.ACTION, shape=(1,)),
    ]

    assert resolve_action_dim(features) == 7
    assert replace(_config(), output_features=features, action_dim=7).action_dim == 7


@pytest.mark.parametrize(
    "features", [[], [Feature(ftype=FeatureType.STATE, shape=(2,))], [Feature(ftype=FeatureType.ACTION, shape=())]]
)
def test_action_dimension_rejects_missing_or_non_action_features(features: list[Feature]) -> None:
    with pytest.raises(ValueError, match="output feature|must be an action"):
        resolve_action_dim(features)


def test_configuration_requires_visual_input_and_matching_action_width() -> None:
    with pytest.raises(ValueError, match="visual input"):
        replace(_config(), input_features=[Feature(ftype=FeatureType.STATE, shape=(2,))])
    with pytest.raises(ValueError, match="action_dim does not match"):
        replace(_config(), action_dim=3)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"n_obs_steps": 0}, "must be positive"),
        ({"chunk_size": 0}, "must be positive"),
        ({"n_action_steps": 0}, "n_action_steps"),
        ({"n_action_steps": 6}, "n_action_steps"),
        ({"hidden_dim": 0}, "must be positive"),
        ({"num_layers": 0}, "must be positive"),
        ({"num_heads": 0}, "must be positive"),
        ({"hidden_dim": 15, "num_heads": 2}, "divisible"),
    ],
)
def test_configuration_rejects_invalid_horizons_and_transformer_shapes(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **changes)


@pytest.mark.parametrize("field", ["encoder_name", "encoder_output", "action_head"])
def test_configuration_rejects_unknown_architectures(field: str) -> None:
    with pytest.raises(ValueError, match=f"Unsupported {field}"):
        replace(_config(), **{field: "unknown"})


def test_webssl_requires_resolved_offline_encoder_metadata() -> None:
    with pytest.raises(ValueError, match="offline reconstruction"):
        replace(_config(), encoder_name="webssl")
    config = replace(_config(), encoder_name="webssl", encoder_config={"architecture": "tiny"})

    assert PatchPolicyConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("changes", [{"vq_num_groups": 0}, {"vq_num_embeddings": 0}, {"vq_action_scale": 0.0}])
def test_configuration_rejects_nonpositive_quantizer_settings(changes: dict) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        replace(_config(), **changes)
