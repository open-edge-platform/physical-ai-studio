# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Patch Policy configuration contract."""

from dataclasses import replace
from inspect import signature
from pathlib import Path

import pytest
import torch
from jsonargparse import ArgumentParser

from physicalai.cli._dispatch import _build_lightning_parser
from physicalai.config import Config
from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.gyms import Gym, HistoryGym, PushTGym
from physicalai.policies.patch_policy.config import PatchPolicyConfig, resolve_action_dim
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.policy import PatchPolicy
from physicalai.train import RolloutVideoRecorderCallback


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


def test_defaults_match_real_robot_recipe(tiny_encoder) -> None:
    config = _config()
    policy = PatchPolicy(encoder_pretrained=False)
    restored = PatchPolicy.from_config(config)
    expected = {
        "encoder_name": "dinov2",
        "action_head": "behavior_transformer",
        "n_obs_steps": 2,
        "chunk_size": 5,
        "n_action_steps": 1,
        "hidden_dim": 384,
        "num_layers": 8,
        "num_heads": 8,
    }
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert getattr(policy, name) == value
        assert getattr(restored, name) == value
        if name != "n_action_steps":
            assert signature(PatchPolicyModel).parameters[name].default == value
    for instance in (policy, restored):
        assert instance.optimizer_lr == 3e-4
        assert instance.optimizer_weight_decay == 1e-4
        assert instance.compile_model is True
        assert instance.vqvae_iters == 1000
        assert instance.vqvae_batch_size == 1024
    assert config.vq_latent_dim == 512


def test_pusht_yaml_preserves_reference_recipe() -> None:
    path = Path(__file__).resolve().parents[4] / "configs/physicalai/patch_policy/pusht/dinov2-vqbet.yaml"
    cfg = _build_lightning_parser("fit").parse_args(["--config", str(path)])
    model = cfg.model.init_args
    assert cfg.model.class_path == "physicalai.policies.PatchPolicy"
    assert model.encoder_name == "dinov2"
    assert model.encoder_pretrained is True
    assert model.action_head == "behavior_transformer"
    assert (model.n_obs_steps, model.chunk_size, model.n_action_steps) == (5, 5, 1)
    assert (model.hidden_dim, model.num_layers, model.num_heads) == (512, 8, 8)
    assert (model.vqvae_iters, model.vqvae_batch_size) == (1000, 1024)
    assert model.optimizer_lr == 5.5e-5
    assert model.optimizer_weight_decay == 2e-4
    assert model.compile_model is True
    assert model.use_random_noise is True
    data = cfg.data.init_args
    assert data.repo_id == "alfieroddan/pushT-224x224"
    assert data.revision == "c50cea16d2f0884353c9bd65113ee9cc892d094f"
    assert (data.train_batch_size, data.num_workers, data.num_rollouts_val) == (8, 0, 10)
    assert data.val_split == 0.0
    trainer = cfg.trainer
    assert trainer.max_epochs == 400
    assert trainer.max_steps == -1
    assert (trainer.accelerator, trainer.devices, trainer.precision) == ("gpu", 1, "32-true")
    assert trainer.accumulate_grad_batches == 1
    assert trainer.num_sanity_val_steps == 1
    assert trainer.check_val_every_n_epoch == 10
    assert trainer.val_check_interval is None
    assert trainer.log_every_n_steps == 100
    assert trainer.logger.class_path == "lightning.pytorch.loggers.WandbLogger"
    assert trainer.logger.init_args.project == "pushT-224x224"
    assert trainer.logger.init_args.name == "dinov2-vqbet-hub224-v2"
    assert trainer.logger.init_args.save_dir == "output"
    callback = trainer.callbacks[0]
    assert callback["class_path"] == "physicalai.train.RolloutVideoRecorderCallback"
    recorder = RolloutVideoRecorderCallback(**callback["init_args"])
    assert recorder.output_dir == Path("output/rollouts")
    assert (recorder.fps, recorder.frame_key, recorder.record_mode) == (10, "top", "all")

    parser = ArgumentParser()
    parser.add_subclass_arguments(Gym, "gym")
    gym = parser.instantiate(parser.parse_object({"gym": data.val_gym.as_dict()})).gym
    try:
        assert isinstance(gym, HistoryGym)
        assert isinstance(gym.gym, PushTGym)
        assert (gym.n_obs_steps, gym.stride) == (5, 1)
        observation, _ = gym.reset(seed=42)
        images = observation.images["top"]
        assert images.shape == (1, 5, 3, 224, 224)
        torch.testing.assert_close(images[:, 0], images[:, -1])
        next_observation, *_ = gym.step(gym.sample_action())
        torch.testing.assert_close(next_observation.images["top"][:, :-1], images[:, 1:])
        assert gym.keys == {"images.top"}
    finally:
        gym.close()


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
