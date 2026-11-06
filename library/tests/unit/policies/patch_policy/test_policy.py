# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Readable construction and Lightning lifecycle checks without downloads."""

from dataclasses import replace
from types import SimpleNamespace

import lightning
import pytest
import torch

from physicalai.data import Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.policies.patch_policy import policy as policy_module
from physicalai.policies.patch_policy.policy import PatchPolicy


class FeatureDataset(Dataset):
    """Only dataset metadata is needed by setup; sample access must not occur."""

    def __init__(self, config):
        self.inputs = {feature.name: feature for feature in config.input_features}
        self.outputs = {feature.name: feature for feature in config.output_features}
        self._delta_indices = {}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("setup must not load samples")

    raw_features = property(lambda self: {})
    observation_features = property(lambda self: self.inputs)
    action_features = property(lambda self: self.outputs)
    fps = property(lambda self: 10)
    tolerance_s = property(lambda self: 0.01)
    delta_indices = property(
        lambda self: self._delta_indices, lambda self, values: setattr(self, "_delta_indices", values)
    )


def test_unresolved_configuration_does_not_build_modules():
    policy = PatchPolicy(encoder_pretrained=False)

    policy.configure_model()

    assert policy.model is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    with pytest.raises(RuntimeError, match="config is not initialized"):
        _ = policy.config
    with pytest.raises(RuntimeError, match="not initialized"):
        policy.predict_action_chunk(Observation())
    with pytest.raises(RuntimeError, match="Cannot configure optimizers"):
        policy.configure_optimizers()


def test_constructor_builds_when_features_are_provided(tiny_encoder, small_config):
    policy = PatchPolicy(
        input_features=small_config.input_features,
        output_features=small_config.output_features,
        encoder_pretrained=False,
        action_head="diffusion",
        n_obs_steps=2,
        chunk_size=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        num_inference_steps=2,
    )

    assert policy.model is not None
    assert policy._preprocessor is not None
    assert policy.config.action_dim == 2


def test_dataset_setup_resolves_features_without_building_modules(tiny_encoder, small_config):
    policy = PatchPolicy(encoder_pretrained=False, action_head="diffusion", hidden_dim=8, num_layers=1, num_heads=2)
    policy.trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=FeatureDataset(small_config)))

    policy.setup("fit")

    assert policy.input_features == small_config.input_features
    assert policy.output_features == small_config.output_features
    assert policy.model is None
    assert policy._preprocessor is None
    policy.configure_model()
    assert policy.model is not None


def test_repeated_configuration_preserves_modules_config_and_queued_actions(tiny_encoder, small_config):
    policy = PatchPolicy.from_config(small_config)
    original = (policy.model, policy.config, policy._preprocessor, policy._postprocessor)
    parameters = tuple(policy.model.parameters())
    queued_action = torch.ones(1, 2)
    policy._action_queue.append(queued_action)

    policy.configure_model()

    assert all(
        before is after
        for before, after in zip(original, (policy.model, policy.config, policy._preprocessor, policy._postprocessor))
    )
    assert all(before is after for before, after in zip(parameters, policy.model.parameters()))
    assert policy._action_queue[0] is queued_action


def test_changed_statistics_refresh_processors_not_parameters(tiny_encoder, small_config):
    policy = PatchPolicy.from_config(small_config)
    model = policy.model
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    statistics = NormalizationParameters(min=[0.0, 0.0], max=[4.0, 4.0])
    policy.output_features = [replace(policy.output_features[0], normalization_data=statistics)]

    policy.configure_model()

    assert policy.model is model
    assert policy._preprocessor is not preprocessor
    assert policy._postprocessor is not postprocessor
    assert policy.config.output_features[0].normalization_data == statistics
    torch.testing.assert_close(policy._postprocessor(torch.ones(1, 2, 2)), torch.full((1, 1, 2), 4.0))


@pytest.mark.parametrize("missing", ["_preprocessor", "_postprocessor"])
def test_missing_processor_is_repaired_without_rebuilding(tiny_encoder, small_config, missing):
    policy = PatchPolicy.from_config(small_config)
    model = policy.model
    config = policy.config
    setattr(policy, missing, None)

    policy.configure_model()

    assert policy.model is model
    assert policy.config is config
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None


def test_incompatible_features_fail_before_synchronized_state_changes(tiny_encoder, small_config):
    policy = PatchPolicy.from_config(small_config)
    config = policy.config
    model = policy.model
    preprocessor = policy._preprocessor
    policy.input_features = [Feature(name="other_camera", ftype=FeatureType.VISUAL, shape=(3, 4, 4))]

    with pytest.raises(ValueError, match="does not match"):
        policy.configure_model()

    assert policy.config is config
    assert policy.model is model
    assert policy._preprocessor is preprocessor
    assert policy.model.camera_names == ("front",)


@pytest.mark.parametrize("change", ["architecture", "camera_order"])
def test_incompatible_checkpoint_is_rejected_before_mutation(tiny_encoder, small_config, change):
    second_camera = Feature(name="back", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    config = replace(small_config, input_features=[*small_config.input_features, second_camera])
    policy = PatchPolicy.from_config(config)
    saved_config = policy.config
    if change == "architecture":
        incompatible = replace(saved_config, hidden_dim=16)
    else:
        incompatible = replace(saved_config, input_features=list(reversed(saved_config.input_features)))

    with pytest.raises(ValueError, match="does not match"):
        policy.on_load_checkpoint({"policy_config": incompatible.to_dict()})

    assert policy.config is saved_config
    assert policy.input_features is saved_config.input_features
    assert policy.model.camera_names == ("front", "back")


def test_checkpoint_round_trip_restores_weights_statistics_and_execution_horizon(
    tiny_encoder, small_config, tmp_path, monkeypatch
):
    config = replace(small_config, n_action_steps=2)
    policy = PatchPolicy.from_config(config)
    with torch.no_grad():
        next(policy.model.action_head.parameters()).fill_(0.25)
    checkpoint = {
        "state_dict": policy.state_dict(),
        "hyper_parameters": dict(policy.hparams),
        "pytorch-lightning_version": lightning.__version__,
    }
    policy.on_save_checkpoint(checkpoint)
    path = tmp_path / "policy.ckpt"
    torch.save(checkpoint, path)

    def forbid_download(*args, **kwargs):
        raise AssertionError("checkpoint restoration must not download encoder artifacts")

    monkeypatch.setattr(policy_module, "resolve_encoder_artifacts", forbid_download)
    restored = PatchPolicy.load_from_checkpoint(path, n_action_steps=1)

    assert restored.config == policy.config
    assert restored._n_action_steps == 2
    assert restored._action_queue.maxlen == 2
    assert not restored.encoder_pretrained
    for name, expected in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_training_and_prediction_use_processors(tiny_encoder, small_config, monkeypatch):
    policy = PatchPolicy.from_config(small_config)
    observation = Observation(images={"front": torch.zeros(1, 2, 3, 4, 4)}, action=torch.ones(1, 3, 2))
    logged = {}
    monkeypatch.setattr(policy, "log_dict", lambda metrics, **kwargs: logged.update(metrics))

    loss = policy.training_step(observation, 0)
    loss.backward()
    policy.eval()
    monkeypatch.setattr(policy.model.ema_action_head, "sample", lambda context: torch.zeros(1, 2, 2))
    actions = policy.predict_action_chunk(observation)

    assert torch.isfinite(loss)
    assert "train/loss" in logged
    torch.testing.assert_close(actions, torch.ones(1, 1, 2))
    with pytest.raises(TypeError, match="Training requires tensor actions"):
        policy._prepare_batch(Observation(images=observation.images), require_actions=True)


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_optimizer_contains_only_trainable_parameters(tiny_encoder, small_config, action_head):
    policy = PatchPolicy.from_config(replace(small_config, action_head=action_head))

    optimizer = policy.configure_optimizers()

    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = [parameter for parameter in policy.model.parameters() if parameter.requires_grad]
    assert len(actual) == len({id(parameter) for parameter in actual})
    assert {id(parameter) for parameter in actual} == {id(parameter) for parameter in expected}
    if action_head == "behavior_transformer":
        assert [group["weight_decay"] for group in optimizer.param_groups] == [2e-4, 0.0, 0.01, 2e-4]


def test_encoder_weights_and_compilation_are_applied_once(tiny_encoder, small_config, monkeypatch):
    monkeypatch.setattr(
        policy_module, "resolve_encoder_artifacts", lambda *args, **kwargs: (None, {"weight": torch.tensor(3.0)})
    )
    policy = PatchPolicy(
        small_config.input_features,
        small_config.output_features,
        action_head="diffusion",
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        compile_model=True,
    )

    policy.configure_model()

    assert policy.model.encoder.weight_loads == 1
    assert policy.model.encoder.compile_calls == 1
    torch.testing.assert_close(policy.model.encoder.weight, torch.tensor(3.0))
    assert policy._encoder_weights is None
