# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the modular Patch Policy pipeline."""

from __future__ import annotations

import warnings
from dataclasses import fields, replace
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import lightning as L  # noqa: N812
import numpy as np
import pytest
import torch

import physicalai.policies.patch_policy as patch_policy
from physicalai.data import DataModule, Dataset, Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.observation import ACTION, IMAGES
from physicalai.export import ExportBackend
from physicalai.inference.adapters.pytorch import TorchAdapter
from physicalai.inference.manifest import Manifest
from physicalai.inference.preprocessors import (
    PatchPolicyPreprocessor as NumpyPatchPolicyPreprocessor,
    ResizePreprocessor,
    StatsNormalizer,
    ToFloatTensorPreprocessor,
)
from physicalai.policies.base import Policy
from physicalai.policies.patch_policy import PatchPolicy
from physicalai.policies.patch_policy.components.encoder import TimmFeatureEncoder
from physicalai.policies.patch_policy.components.heads import BehaviourTransformerHead
from physicalai.policies.patch_policy.config import FeatureEncoderName, PatchPolicyConfig
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.policy import _copy_normalization_data
from physicalai.policies.patch_policy.processors import PatchPolicyPostprocessor, PatchPolicyPreprocessor
from physicalai.policies.utils import in_episode_bound, reduce_losses
from physicalai.train import Trainer


def features() -> tuple[list[Feature], list[Feature]]:
    """Return a minimal visual/action contract."""
    return (
        [Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 32, 32))],
        [Feature(name="action", ftype=FeatureType.ACTION, shape=(2,))],
    )


def make_policy(
    action_head: str, *, use_random_input_noise: bool = True, image_size: tuple[int, int] = (224, 224),
    n_action_steps: int = 2,
) -> PatchPolicy:
    """Construct a small policy without downloading pretrained weights."""
    inputs, outputs = features()
    policy = PatchPolicy(
        input_features=inputs,
        output_features=outputs,
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
        image_size=image_size,
        action_head=action_head,  # type: ignore[arg-type]
        n_obs_steps=2,
        chunk_size=4,
        n_action_steps=n_action_steps,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        num_inference_steps=2,
        use_random_input_noise=use_random_input_noise,
    )
    if isinstance(policy.model.action_head, BehaviourTransformerHead):
        policy.model.action_head.quantizer.fitted.fill_(True)
    return policy


def batch() -> Observation:
    """Return a synthetic two-step visual batch."""
    return Observation(
        images={"front": torch.rand(2, 2, 3, 32, 32)},
        action=torch.rand(2, 5, 2) * 2 - 1,
    )


def test_package_exports_only_public_policy_types() -> None:
    """The package exports config, model, and policy without private components."""
    assert patch_policy.__all__ == ["PatchPolicy", "PatchPolicyConfig", "PatchPolicyModel"]
    assert not hasattr(patch_policy, "EmbeddingMemoryBank")


def test_default_settings_match_pusht_vqbet() -> None:
    inputs, outputs = features()
    config = PatchPolicyConfig(input_features=inputs, output_features=outputs, action_dim=2)
    expected = {
        "action_head": "behavior_transformer",
        "encoder_name": "dinov2",
        "n_obs_steps": 5,
        "chunk_size": 5,
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "dropout": 0.0,
        "vq_latent_dim": 512,
        "vq_action_scale": 500.0,
    }
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert signature(PatchPolicy).parameters[name].default == value
        assert signature(PatchPolicyModel).parameters[name].default == value
    assert config.n_action_steps == signature(PatchPolicy).parameters["n_action_steps"].default == 1
    for name, value in {"optimizer_lr": 5.5e-5, "optimizer_weight_decay": 2e-4}.items():
        assert signature(PatchPolicy).parameters[name].default == value
        assert signature(PatchPolicy.from_config).parameters[name].default == value


def test_optimizer_configuration_does_not_prepare_training(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    policy.trainer = Mock()
    preparation = Mock(side_effect=AssertionError("Optimizer creation must not prepare training"))
    monkeypatch.setattr("physicalai.policies.patch_policy.policy.fit_tokenizer", preparation)
    optimizer = policy.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    preparation.assert_not_called()


def test_fit_start_calls_parent_before_tokenizer_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    calls = []
    monkeypatch.setattr(Policy, "on_fit_start", lambda _self: calls.append("parent"))
    monkeypatch.setattr("physicalai.policies.patch_policy.policy.fit_tokenizer", lambda _policy: calls.append("tokenizer"))

    policy.on_fit_start()

    assert calls == ["parent", "tokenizer"]


@pytest.mark.parametrize("compile_model", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_compile_feature_encoder_once_without_saving_option(
    monkeypatch: pytest.MonkeyPatch, compile_model: bool, deferred: bool,
) -> None:
    compile_encoder = Mock()
    monkeypatch.setattr(TimmFeatureEncoder, "compile", compile_encoder)
    inputs, outputs = features()
    policy = PatchPolicy(
        input_features=None if deferred else inputs,
        output_features=None if deferred else outputs,
        encoder_name="resnet18", encoder_output="average", encoder_pretrained=False,
        hidden_dim=8, num_layers=1, num_heads=2, compile_model=compile_model,
    )
    if deferred:
        compile_encoder.assert_not_called()
        policy.input_features, policy.output_features = inputs, outputs
        policy.configure_model()
    encoder = policy.model.encoder
    keys = set(policy.state_dict())
    policy.configure_model()
    policy.configure_model()
    assert compile_encoder.call_count == int(compile_model)
    assert policy.model.encoder is encoder
    assert set(policy.state_dict()) == keys
    assert "compile_model" not in policy.hparams
    assert "compile_model" not in policy.config.to_dict()


def test_behavior_transformer_optimizer_matches_upstream_groups() -> None:
    policy = make_policy("behavior_transformer")
    optimizer = policy.configure_optimizers()
    head = policy.model.action_head
    decay_by_parameter = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert len(decay_by_parameter) == sum(parameter.requires_grad for parameter in policy.parameters())
    for module in head.trunk.modules():
        for name, parameter in module.named_parameters(recurse=False):
            expected = policy.optimizer_weight_decay if isinstance(module, torch.nn.Linear) and name == "weight" else 0.0
            assert decay_by_parameter[id(parameter)] == expected
    for parameter in head.code_head.parameters():
        assert decay_by_parameter[id(parameter)] == 0.01
    for parameter in head.offset_head.parameters():
        assert decay_by_parameter[id(parameter)] == policy.optimizer_weight_decay
    assert all(group["lr"] == 5.5e-5 and group["betas"] == (0.9, 0.999) for group in optimizer.param_groups)


@pytest.mark.parametrize("n_obs_steps", [0, -1])
def test_preprocessor_requires_positive_observation_window(n_obs_steps: int) -> None:
    with pytest.raises(ValueError, match="n_obs_steps must be positive"):
        PatchPolicyPreprocessor([], [], (4, 4), n_obs_steps=n_obs_steps)


@pytest.mark.parametrize("flatten", [False, True])
def test_preprocessor_preserves_action_padding(flatten: bool) -> None:
    inputs, outputs = features()
    processor = PatchPolicyPreprocessor(inputs, outputs, (4, 4), n_obs_steps=2)
    observation = batch()
    padding = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])
    observation.extra = {"action_is_pad": padding}

    processed = processor(observation.to_dict(flatten=flatten))

    torch.testing.assert_close(processed["extra.action_is_pad"], padding)
    torch.testing.assert_close(in_episode_bound(processed), ~padding)
    losses = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]]).expand(2, -1, -1)
    torch.testing.assert_close(reduce_losses(losses, in_episode_bound(processed)), torch.tensor(2.0))


def test_behavior_transformer_loss_has_action_and_model_gradients() -> None:
    torch.manual_seed(0)
    head = BehaviourTransformerHead(
        8,
        2,
        4,
        n_obs_steps=1,
        n_patches=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        latent_dim=4,
        num_embeddings=4,
        groups=2,
    ).eval()
    actions = torch.randn(2, 4, 2, requires_grad=True)
    context = {"tokens": torch.zeros(2, 1, 8)}

    loss = head.compute_loss(actions, context)
    loss.backward()

    assert torch.isfinite(loss)
    assert actions.grad is not None
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_behavior_transformer_unpadded_loss_is_unchanged() -> None:
    head = BehaviourTransformerHead(
        8,
        2,
        4,
        n_obs_steps=1,
        n_patches=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        latent_dim=4,
        num_embeddings=4,
        groups=2,
    ).eval()
    actions = torch.randn(2, 4, 2)
    context = {"tokens": torch.zeros(2, 1, 8)}

    torch.manual_seed(7)
    without_mask = head.compute_loss(actions, context)
    torch.manual_seed(7)
    with_mask = head.compute_loss(actions, context, valid_steps=torch.ones(2, 4, dtype=torch.bool))

    torch.testing.assert_close(with_mask, without_mask)


def test_behavior_transformer_backward_updates_trunk() -> None:
    """Transformer residual connections preserve gradients during a full backward pass."""
    head = BehaviourTransformerHead(
        8,
        2,
        4,
        n_obs_steps=2,
        n_patches=2,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        latent_dim=4,
        num_embeddings=4,
        groups=2,
    )
    actions = torch.randn(2, 5, 2)
    context = {"tokens": torch.randn(2, 4, 8)}

    head.quantizer.fitted.fill_(True)
    head.compute_loss(actions, context).mean().backward()

    assert any(parameter.grad is not None for parameter in head.trunk.parameters())


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_model_training_and_validation_use_action_padding(action_head: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy(action_head)
    model = policy.model
    assert isinstance(model, PatchPolicyModel)
    observation = batch()
    padding = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])
    observation.extra = {"action_is_pad": padding}
    prepared = policy._prepare_batch(observation, require_actions=True)
    losses = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]]).expand(2, -1, -1)
    compute_loss = Mock(return_value=torch.tensor(2.0) if action_head == "behavior_transformer" else losses)
    monkeypatch.setattr(model, "_context", Mock(return_value={}))
    monkeypatch.setattr(model.action_head, "compute_loss", compute_loss)
    target = prepared["action"][:, 1:]
    sampling_head = model.ema_action_head or model.action_head
    monkeypatch.setattr(sampling_head, "sample", Mock(return_value=target + torch.sqrt(losses[:, 1:])))

    training_loss, _ = model.compute_loss(prepared)
    validation_loss, _ = model.compute_val_loss(prepared)

    torch.testing.assert_close(training_loss, torch.tensor(2.0))
    torch.testing.assert_close(validation_loss, torch.tensor(3.0))
    if action_head == "behavior_transformer":
        torch.testing.assert_close(compute_loss.call_args.kwargs["valid_steps"], ~padding)
    else:
        assert not compute_loss.call_args.kwargs


def test_processors_warn_when_action_normalization_data_is_missing() -> None:
    _, outputs = features()

    with pytest.warns(UserWarning, match="normalization"):
        PatchPolicyPreprocessor([], outputs, (32, 32))
    with pytest.warns(UserWarning, match="de-normalization"):
        PatchPolicyPostprocessor(outputs, n_action_steps=1)


@torch.no_grad()
def test_processors_normalize_images_and_concatenated_actions() -> None:
    """Camera features match by name regardless of mapping order."""
    inputs = [
        Feature(
            name="front",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[0.5, 1.0, 1.5], std=[0.5, 1.0, 0.5]),
        ),
        Feature(
            name="wrist",
            ftype=FeatureType.VISUAL,
            shape=(3, 4, 4),
            normalization_data=NormalizationParameters(mean=[1.0, 1.0, 1.0], std=[2.0, 2.0, 2.0]),
        ),
    ]
    outputs = [
        Feature(
            name="action.arm",
            ftype=FeatureType.ACTION,
            shape=(2,),
            normalization_data=NormalizationParameters(mean=[1.0, 2.0], std=[2.0, 4.0]),
        ),
        Feature(
            name="action.gripper",
            ftype=FeatureType.ACTION,
            shape=(1,),
            normalization_data=NormalizationParameters(mean=[3.0], std=[0.5]),
        ),
    ]
    preprocessor = PatchPolicyPreprocessor(inputs, outputs, (4, 4))
    postprocessor = PatchPolicyPostprocessor(outputs, n_action_steps=1)
    raw_actions = torch.tensor([[[3.0, 6.0, 3.5], [1.0, 2.0, 3.0]]])

    processed = preprocessor(
        {
            "images": {
                "wrist": torch.full((1, 1, 3, 8, 8), 3.0),
                "front": torch.cat(
                    [torch.full((1, 1, 1, 8, 8), value) for value in (0.5, 1.0, 2.0)],
                    dim=2,
                ),
            },
            "action": raw_actions,
        },
    )
    flattened = preprocessor(
        {
            "images.wrist": torch.full((1, 1, 3, 8, 8), 3.0),
            "images.front": torch.cat(
                [torch.full((1, 1, 1, 8, 8), value) for value in (0.5, 1.0, 2.0)],
                dim=2,
            ),
            "action": raw_actions,
        },
    )

    assert processed["images.front"].shape == (1, 1, 3, 4, 4)
    assert processed["images.wrist"].shape == (1, 1, 3, 4, 4)
    torch.testing.assert_close(processed["images.front"][0, 0, :, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(processed["images.wrist"][0, 0, :, 0, 0], torch.ones(3))
    assert processed["action"].shape == (1, 2, 3)
    torch.testing.assert_close(processed["action"], torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]))
    torch.testing.assert_close(flattened["images.front"], processed["images.front"])
    torch.testing.assert_close(flattened["images.wrist"], processed["images.wrist"])
    torch.testing.assert_close(flattened["action"], processed["action"])
    torch.testing.assert_close(postprocessor(processed["action"]), raw_actions[:, :1])


def test_preprocessor_adds_singleton_time_axis_to_single_frame() -> None:
    """Single frames gain a time axis without processor-owned history or padding."""
    feature = Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4), n_obs_steps=1)
    frame = torch.full((1, 3, 4, 4), 5.0)

    processed = preprocessor({"images": {"front": frame}})

    assert processed["images.front"].shape == (1, 1, 3, 4, 4)
    torch.testing.assert_close(processed["images.front"][0, 0, :, 0, 0], torch.full((3,), 5.0))
    torch.testing.assert_close(preprocessor({"images.front": frame[:, None]})["images.front"], processed["images.front"])


def test_preprocessor_uses_mean_std_feature_statistics() -> None:
    """Patch Policy applies mean/std normalization when feature statistics exist."""
    feature = Feature(
        name="front",
        ftype=FeatureType.VISUAL,
        shape=(3, 4, 4),
        normalization_data=NormalizationParameters(mean=[1.0, 1.0, 1.0], std=[2.0, 2.0, 2.0]),
    )
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4))
    image = torch.full((1, 3, 4, 4), 3.0)

    processed = preprocessor({"images": {"front": image}})["images.front"]

    torch.testing.assert_close(processed, torch.ones_like(image[:, None]))


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("timesteps", [1, 3])
@pytest.mark.parametrize("image_layout", ["BCHW", "BHWC"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_numpy_image_pipeline_matches_torch(
    flatten: bool,
    timesteps: int,
    image_layout: str,
    dtype: type[np.uint8] | type[np.float32] | type[np.float64],
) -> None:
    """Shared NumPy stages match Torch image preparation without action processing."""
    mean = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    std = np.array([0.5, 0.75, 1.0], dtype=np.float32)
    inputs = [
        Feature(
            name="wrist", ftype=FeatureType.VISUAL, shape=(3, 5, 7),
            normalization_data=NormalizationParameters(mean=mean.tolist(), std=std.tolist()),
        ),
        Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 5, 7)),
    ]
    torch_processor = PatchPolicyPreprocessor(inputs, [], (8, 10), n_obs_steps=timesteps)
    camera_processor = NumpyPatchPolicyPreprocessor(["wrist", "front"], n_obs_steps=3)
    resizer = ResizePreprocessor((8, 10), image_layout=image_layout, mode="stretch")
    normalizer = StatsNormalizer(stats={
        f"{IMAGES}.wrist": {"mean": mean.reshape(3, 1, 1), "std": std.reshape(3, 1, 1)},
    })
    rng = np.random.default_rng(42)
    images: dict[str, np.ndarray] = {
        name: rng.integers(0, 256, (2, timesteps, 3, 5, 7), dtype=np.uint8)
        for name in ("front", "wrist")
    }
    if dtype != np.uint8:
        images = {name: image.astype(dtype) / 255.0 for name, image in images.items()}
    torch_images = {name: torch.from_numpy(image) for name, image in images.items()}
    if image_layout == "BHWC":
        images = {name: image.transpose(0, 1, 3, 4, 2) for name, image in images.items()}
    numpy_batch = {f"{IMAGES}.{name}": image for name, image in images.items()} if flatten else {IMAGES: images}
    torch_batch = (
        {f"{IMAGES}.{name}": image for name, image in torch_images.items()}
        if flatten else {IMAGES: torch_images}
    )

    prepared = camera_processor(numpy_batch)
    resized = resizer(ToFloatTensorPreprocessor()(prepared))
    actual = normalizer({key: np.asarray(value) for key, value in resized.items()})
    expected = torch_processor(torch_batch)

    assert list(actual) == list(expected) == [f"{IMAGES}.wrist", f"{IMAGES}.front"]
    for key in expected:
        assert actual[key].dtype == np.float32
        assert actual[key].shape == (2, timesteps, 3, 8, 10)
        np.testing.assert_allclose(actual[key], expected[key].numpy(), atol=1e-6, rtol=1e-5)


def test_preprocessor_scales_uint8_before_image_normalization() -> None:
    """Uint8 pixels are converted to [0, 1] before ImageNet normalization."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    feature = Feature(
        name="front",
        ftype=FeatureType.VISUAL,
        shape=(3, 4, 4),
        normalization_data=NormalizationParameters(mean=mean, std=std),
    )
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4))
    image = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)

    processed = preprocessor({"images": {"front": image}})["images.front"]

    torch.testing.assert_close(processed[0, 0, :, 0, 0], (1 - torch.tensor(mean)) / torch.tensor(std))


def test_policy_attaches_encoder_stats_and_preserves_dataset_action_stats() -> None:
    """Visual features use encoder stats while action features keep dataset stats."""
    policy = make_policy("diffusion")
    visual = policy.config.input_features[0]
    assert visual.normalization_data is not None
    assert visual.normalization_data.mean == pytest.approx([0.485, 0.456, 0.406])
    assert visual.normalization_data.std == pytest.approx([0.229, 0.224, 0.225])

    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    action_feature = Feature(name="action", ftype=FeatureType.ACTION, shape=(2,))
    dataset_feature = Feature(
        name="action",
        ftype=FeatureType.ACTION,
        shape=(2,),
        normalization_data=action_stats,
    )

    resolved = _copy_normalization_data([action_feature], [dataset_feature])

    assert resolved[0].normalization_data == action_stats


def test_setup_resolves_features_from_training_dataset() -> None:
    inputs, outputs = features()
    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    dataset_outputs = [replace(outputs[0], normalization_data=action_stats)]
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in dataset_outputs}
    trainer = Mock(callbacks=[])
    trainer.datamodule.train_dataset = train_dataset
    policy = PatchPolicy(
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
    )
    policy._trainer = trainer

    policy.setup("fit")

    assert policy.model is None
    assert policy.config is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    assert policy.input_features == inputs
    assert policy.output_features == dataset_outputs

    policy.configure_model()

    assert policy.config is not None
    assert isinstance(policy.model, PatchPolicyModel)
    assert policy.input_features == policy.config.input_features
    assert policy.output_features == dataset_outputs
    assert policy.output_features[0].normalization_data == action_stats


def test_configure_model_is_idempotent() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    config = policy.config
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor

    policy.configure_model()

    assert policy.model is model
    assert policy.config is config
    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor


def test_configure_model_defers_until_features_are_resolved() -> None:
    policy = PatchPolicy()

    policy.configure_model()

    assert policy.model is None
    assert policy.config is None
    assert policy._preprocessor is None
    assert policy._postprocessor is None
    with pytest.raises(RuntimeError, match="features and model are resolved"):
        policy.configure_optimizers()


@pytest.mark.parametrize("dataset_stats", [None, NormalizationParameters(mean=[10.0, 20.0], std=[2.0, 2.0])])
def test_setup_replaces_existing_action_statistics(dataset_stats: NormalizationParameters | None) -> None:
    policy = make_policy("diffusion")
    assert policy.output_features is not None
    old_stats = NormalizationParameters(mean=[1.0, 2.0], std=[1.0, 1.0])
    policy.output_features = [replace(policy.output_features[0], normalization_data=old_stats)]
    policy.configure_model()
    model = policy.model
    inputs, outputs = features()
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {"action": replace(outputs[0], normalization_data=dataset_stats)}
    trainer = Mock(callbacks=[])
    trainer.datamodule.train_dataset = train_dataset
    policy.trainer = trainer

    with pytest.warns(UserWarning, match="Replacing normalization statistics"):
        policy.setup("fit")
    policy.configure_model()

    assert policy.model is model
    assert policy.config is not None
    assert policy.config.output_features[0].normalization_data == dataset_stats
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.output_features == policy.config.output_features
    assert policy._postprocessor.output_features == policy.config.output_features
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        policy.setup("fit")
        policy.configure_model()
    assert not captured


def test_configure_model_refreshes_processors_without_rebuilding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert model is not None
    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    monkeypatch.setattr(PatchPolicyModel, "from_config", Mock(side_effect=AssertionError("Model was rebuilt.")))
    inputs, outputs = features()
    action_stats = NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5])
    dataset_outputs = [replace(outputs[0], normalization_data=action_stats)]
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in dataset_outputs}
    trainer = Mock(callbacks=[])
    trainer.datamodule.train_dataset = train_dataset
    policy._trainer = trainer

    policy.setup("fit")

    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor
    policy.configure_model()

    assert policy.model is model
    assert policy.config is not None
    assert policy.config.output_features == dataset_outputs
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.image_size == model.encoder.image_size
    assert policy._preprocessor is not preprocessor
    assert policy._postprocessor is not postprocessor
    assert policy.input_features is not None
    assert policy.input_features[0].normalization_data is not None
    assert policy.input_features[0].normalization_data.mean == pytest.approx([0.485, 0.456, 0.406])

    preprocessor = policy._preprocessor
    postprocessor = policy._postprocessor
    policy.configure_model()
    assert policy.model is model
    assert policy._preprocessor is preprocessor
    assert policy._postprocessor is postprocessor


@pytest.mark.parametrize("feature_kind", ["input", "output"])
def test_setup_rejects_mismatched_explicit_features(feature_kind: str) -> None:
    inputs, outputs = features()
    train_dataset = Mock(spec=Dataset)
    train_dataset.observation_features = {feature.name: feature for feature in inputs}
    train_dataset.action_features = {feature.name: feature for feature in outputs}
    trainer = Mock(callbacks=[])
    trainer.datamodule.train_dataset = train_dataset
    if feature_kind == "input":
        policy = PatchPolicy(input_features=[replace(inputs[0], shape=(3, 16, 16))])
        error = "input_features"
    else:
        policy = PatchPolicy(output_features=[replace(outputs[0], shape=(3,))])
        error = "output_features"
    policy._trainer = trainer

    with pytest.raises(ValueError, match=error):
        policy.setup("fit")


def test_encoder_does_not_apply_image_normalization_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The encoder passes processor-normalized inputs directly to timm."""
    model = make_policy("diffusion").model
    assert isinstance(model, PatchPolicyModel)
    images = torch.rand(1, 1, 3, 32, 32)
    received: list[torch.Tensor] = []

    def capture(value: torch.Tensor) -> torch.Tensor:
        received.append(value)
        return torch.zeros(value.shape[0], model.encoder.output_dim, 1, 1)

    monkeypatch.setattr(model.encoder.backbone, "forward", capture)
    model.encoder(images)

    torch.testing.assert_close(received[0], images.reshape(-1, 3, 32, 32))


def test_preprocessor_requires_matching_temporal_length() -> None:
    """Explicit temporal images stay 5D and must match n_obs_steps."""
    feature = Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4), n_obs_steps=3)
    images = torch.zeros(1, 3, 3, 4, 4)

    processed = preprocessor({"images": {"front": images}})

    assert processed["images.front"].shape == (1, 3, 3, 4, 4)

    for timesteps in (0, 1, 2, 4):
        with pytest.raises(ValueError, match="Expected 3 observation timestep"):
            preprocessor({"images": {"front": torch.zeros(1, timesteps, 3, 4, 4)}})


@pytest.mark.parametrize("shape", [(1, 3, 4, 4), (3, 4, 4), (1, 2, 1, 3, 4, 4)])
def test_preprocessor_rejects_invalid_image_shape(shape: tuple[int, ...]) -> None:
    feature = Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 4, 4))
    preprocessor = PatchPolicyPreprocessor([feature], [], (4, 4), n_obs_steps=2)

    with pytest.raises(ValueError, match="Expected image shape"):
        preprocessor({"images": {"front": torch.zeros(shape)}})


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_select_action_uses_newest_chunk_without_averaging(action_head: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy(action_head, n_action_steps=1).eval()
    chunks = torch.arange(8 * 2 * 4 * 2, dtype=torch.float32).reshape(8, 2, 4, 2)
    predict = Mock(side_effect=list(chunks) + [chunks[0]])
    monkeypatch.setattr(policy.model, "predict_action_chunk", predict)
    for chunk in chunks:
        actual = policy.select_action(batch())
        torch.testing.assert_close(actual, chunk[:, 0])
        assert not policy._action_queue

    assert predict.call_count == len(chunks)
    policy.reset()
    torch.testing.assert_close(policy.select_action(batch()), chunks[0, :, 0])
    assert predict.call_count == len(chunks) + 1


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_select_action_does_not_track_queued_observations(action_head: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy does not preprocess or retain observations while actions are queued."""
    policy = make_policy(action_head)
    policy.eval()
    captured: list[torch.Tensor] = []

    def predict(batch: dict[str, Any]) -> torch.Tensor:
        captured.append(batch["images.front"].clone())
        return torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + 100 * len(captured)

    assert policy.model is not None
    monkeypatch.setattr(policy.model, "predict_action_chunk", predict)
    for value, expected in ((1.0, 100.0), (2.0, 102.0), (3.0, 200.0)):
        action = policy.select_action(Observation(images={"front": torch.full((1, 2, 3, 32, 32), value)}))
        torch.testing.assert_close(action, torch.tensor([[expected, expected + 1]]))

    assert len(captured) == 2
    assert captured[0].shape == (1, 2, 3, *policy.model.encoder.image_size)
    assert captured[1].shape == (1, 2, 3, *policy.model.encoder.image_size)
    mean, std = torch.tensor(0.485), torch.tensor(0.229)
    torch.testing.assert_close(captured[0][0, 0, 0, 0, 0], (torch.tensor(1.0) - mean) / std)
    torch.testing.assert_close(captured[1][0, 0, 0, 0, 0], (torch.tensor(3.0) - mean) / std)

    policy.reset()
    assert not policy._action_queue
    action = policy.select_action(Observation(images={"front": torch.full((1, 2, 3, 32, 32), 4.0)}))
    torch.testing.assert_close(action, torch.tensor([[300.0, 301.0]]))
    assert len(captured) == 3
    torch.testing.assert_close(captured[2][0, 0, 0, 0, 0], (torch.tensor(4.0) - mean) / std)


def test_model_stacks_camera_windows_without_retaining_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete windows preserve temporal/camera ordering independently of prior calls."""
    inputs, outputs = features()
    inputs.append(replace(inputs[0], name="wrist"))
    policy = PatchPolicy(
        input_features=inputs,
        output_features=outputs,
        encoder_name="resnet18",
        encoder_output="average",
        encoder_pretrained=False,
        action_head="diffusion",
        n_obs_steps=2,
        chunk_size=2,
        n_action_steps=1,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        num_train_timesteps=2,
        num_inference_steps=1,
    ).eval()
    model = policy.model
    assert isinstance(model, PatchPolicyModel)
    received: list[torch.Tensor] = []

    def encode(images: torch.Tensor) -> torch.Tensor:
        received.append(images.clone())
        values = images.mean(dim=(-3, -2, -1), keepdim=False)
        return values[..., None, None].expand(*values.shape, 1, model.encoder.output_dim)

    monkeypatch.setattr(model.encoder, "forward", encode)

    sample = {
        "images": {
            "wrist": torch.tensor([2.0, 4.0]).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 4, 4),
            "front": torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1, 1).expand(1, 2, 3, 4, 4),
        },
    }
    first = model._context(sample)
    model._context({"images": {name: torch.zeros_like(image) for name, image in sample["images"].items()}})
    repeated = model._context(sample)

    assert received[0].shape == (1, 2, 2, 3, 4, 4)
    torch.testing.assert_close(received[0][0, 0, 0], torch.ones(3, 4, 4))
    torch.testing.assert_close(received[0][0, 0, 1], torch.full((3, 4, 4), 2.0))
    assert first["tokens"].shape == (1, 4, model.encoder.output_dim)
    torch.testing.assert_close(first["tokens"][0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(repeated["tokens"], first["tokens"])


@pytest.mark.parametrize("flatten", [False, True])
def test_model_camera_order_follows_input_features(flatten: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured feature order, not dictionary or alphabetical order, controls stacking."""
    inputs, _ = features()
    config = replace(
        make_policy("diffusion").config,
        input_features=[replace(inputs[0], name="wrist"), inputs[0]],
    )
    model = PatchPolicyModel.from_config(config).eval()
    images = {"front": torch.ones(1, 2, 3, 4, 4), "wrist": torch.full((1, 2, 3, 4, 4), 2.0)}
    sample = {f"images.{name}": value for name, value in images.items()} if flatten else {"images": images}
    encode = Mock(return_value=torch.zeros(1, 2, 2, 1, model.encoder.output_dim))
    monkeypatch.setattr(model.encoder, "forward", encode)

    model._encode_observations(sample)

    assert model.camera_names == ("wrist", "front")
    torch.testing.assert_close(encode.call_args.args[0], torch.stack([images["wrist"], images["front"]], dim=2))


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
@pytest.mark.parametrize("training", [False, True])
def test_model_requires_complete_temporal_window(
    action_head: str, training: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training and inference both require complete caller-provided history."""
    policy = make_policy(action_head)
    model = policy.model
    assert isinstance(model, PatchPolicyModel)

    def encode(images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*images.shape[:3], 1, model.encoder.output_dim)

    monkeypatch.setattr(model.encoder, "forward", encode)
    model.train(training)

    context = model._context({"images": {"front": torch.zeros(1, 2, 3, 4, 4)}})
    assert context["tokens"].shape == (1, 2, model.encoder.output_dim)
    with pytest.raises(ValueError, match="Expected 2 observation steps"):
        model._context({"images": {"front": torch.zeros(1, 1, 3, 4, 4)}})


def test_camera_dictionary_matches_legacy_6d_context_and_loss() -> None:
    """Dictionary inputs preserve the former internally stacked training computation."""
    policy = make_policy("diffusion")
    model = policy.model
    assert isinstance(model, PatchPolicyModel)
    model.train()
    prepared = policy._prepare_batch(batch(), require_actions=True)
    stacked = torch.stack([prepared[f"images.{name}"] for name in model.camera_names], dim=2)

    expected_context = {"tokens": model.encoder(stacked).flatten(1, 3)}
    actual_context = model._context(prepared)

    torch.testing.assert_close(actual_context["tokens"], expected_context["tokens"])

    torch.manual_seed(17)
    expected_losses = model.action_head.compute_loss(prepared["action"], expected_context)
    expected_loss = reduce_losses(expected_losses, in_episode_bound(prepared))
    torch.manual_seed(17)
    actual_loss, _ = model.compute_loss(prepared)

    torch.testing.assert_close(actual_loss, expected_loss)


@torch.no_grad()
def test_diffusion_shapes() -> None:
    """Diffusion computes a scalar loss and an execution chunk."""
    policy = make_policy("diffusion")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


@torch.no_grad()
def test_behavior_transformer_shapes() -> None:
    """Behavior Transformer computes a scalar loss and an execution chunk."""
    policy = make_policy("behavior_transformer")
    policy.train()
    loss, _ = policy(batch())
    assert loss.shape == ()
    policy.eval()
    assert policy.predict_action_chunk(batch()).shape == (2, 2, 2)


def test_temporal_indices() -> None:
    """The model requests exactly its configured observation and action windows."""
    model = make_policy("diffusion").model
    assert model is not None
    assert model.observation_delta_indices == [-1, 0]
    assert model.action_delta_indices == [-1, 0, 1, 2, 3]


def test_from_config_reconstructs_explicit_values() -> None:
    """Construction from config preserves every model setting without an argument dictionary."""
    original = make_policy("diffusion", use_random_input_noise=False)
    config = replace(original.config, eta=0.0, clip_sample_range=0.5, use_random_input_noise=False)
    restored = PatchPolicy.from_config(config)

    assert restored.config == config
    assert not hasattr(restored, "_config_values")
    assert isinstance(restored.model, PatchPolicyModel)
    assert getattr(restored.model.action_head, "eta") == 0.0
    assert getattr(restored.model.action_head, "clip_sample_range") == 0.5
    assert getattr(restored.model.action_head, "use_random_input_noise") is False


def test_random_input_noise_propagates_from_policy() -> None:
    """The public policy option reaches the resolved config and diffusion head."""
    policy = make_policy("diffusion", use_random_input_noise=False)

    assert policy.config.use_random_input_noise is False
    assert isinstance(policy.model, PatchPolicyModel)
    assert getattr(policy.model.action_head, "use_random_input_noise") is False


@pytest.mark.parametrize("image_size", [(224, 224), (28, 42)])
@torch.no_grad()
def test_dinov2_uses_policy_image_size(image_size: tuple[int, int]) -> None:
    """Policy resolution controls preprocessing, positional embeddings, and patch counts."""
    original_config = make_policy("diffusion").config
    assert original_config is not None
    config = replace(
        original_config,
        encoder_name="dinov2",
        encoder_output="patches",
        image_size=image_size,
    )
    policy = PatchPolicy.from_config(config)
    model = policy.model
    assert isinstance(model, PatchPolicyModel)
    assert policy.config is not None
    assert policy._preprocessor is not None
    expected_size = image_size
    assert model.encoder.image_size == expected_size
    assert expected_size == getattr(model.encoder.backbone, "patch_embed").img_size
    assert policy._preprocessor.image_size == expected_size
    assert policy.hparams["image_size"] == image_size
    assert "image_size" in policy.config.to_dict()
    prepared = policy._prepare_batch(batch(), require_actions=False)
    assert prepared["images.front"].shape[-2:] == expected_size
    expected_tokens = (expected_size[0] // 14) * (expected_size[1] // 14)

    assert model.encoder.num_tokens == expected_tokens
    tokens = model.encoder(torch.rand(1, 1, 1, 3, *expected_size))
    assert tokens.shape == (1, 1, 1, expected_tokens, model.encoder.output_dim)


def test_encoder_is_frozen_without_tracking_input_gradients() -> None:
    """Encoder features stay constant while downstream heads remain trainable."""
    encoder = TimmFeatureEncoder("resnet18", "patches", pretrained=False)
    encoder.train()
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    head = torch.nn.Linear(encoder.output_dim, 1)

    with torch.enable_grad():
        tokens = encoder(images)
        head(tokens).sum().backward()

    assert not encoder.backbone.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert not tokens.requires_grad
    assert tokens.grad_fn is None
    assert images.grad is None
    assert head.weight.grad is not None


def test_resnet18_uses_fallback_image_size() -> None:
    policy = make_policy("diffusion")

    assert policy.model.encoder.image_size == (224, 224)
    assert policy._preprocessor.image_size == (224, 224)
    prepared = policy._prepare_batch(batch(), require_actions=False)
    assert prepared["images.front"].shape[-2:] == (224, 224)


@pytest.mark.parametrize("encoder_name", ["dinov2", "webssl", "resnet18"])
@pytest.mark.parametrize("image_size", [(224, 224), (28, 42)])
def test_encoders_accept_image_size(
    encoder_name: FeatureEncoderName, image_size: tuple[int, int], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every backbone accepts the default resolution and a rectangular override."""
    from transformers import Dinov2Config  # noqa: PLC0415

    config = Dinov2Config(
        image_size=224,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    monkeypatch.setattr(Dinov2Config, "from_pretrained", Mock(return_value=config))

    encoder = TimmFeatureEncoder(encoder_name, "patches", pretrained=False, image_size=image_size)

    expected_size = image_size
    assert encoder.image_size == expected_size
    expected_tokens = 1 if encoder_name == "resnet18" else (expected_size[0] // 14) * (expected_size[1] // 14)
    assert encoder.num_tokens == expected_tokens
    tokens = encoder(torch.rand(1, 1, 1, 3, *expected_size))
    assert tokens.shape == (1, 1, 1, expected_tokens, encoder.output_dim)


def test_checkpoint_hooks_restore_complete_config() -> None:
    """Checkpoint hooks rebuild a lazy policy from its complete serialized config."""
    original = make_policy("diffusion", use_random_input_noise=False, image_size=(28, 42))
    assert original.config is not None
    checkpoint: dict[str, object] = {}
    original.on_save_checkpoint(checkpoint)

    restored = PatchPolicy(n_action_steps=original.config.n_action_steps)
    restored.on_load_checkpoint(checkpoint)

    assert restored.config == original.config
    assert isinstance(restored.model, PatchPolicyModel)
    assert restored.config is not None
    for field in fields(restored.config):
        if field.name in restored.hparams:
            assert getattr(restored, field.name) == getattr(restored.config, field.name)
            assert restored.hparams[field.name] == getattr(restored.config, field.name)


@pytest.mark.parametrize("processor_name", ["_preprocessor", "_postprocessor"])
def test_checkpoint_repairs_existing_model_processors(processor_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = make_policy("diffusion")
    checkpoint: dict[str, Any] = {}
    policy.on_save_checkpoint(checkpoint)
    model = policy.model
    setattr(policy, processor_name, None)
    monkeypatch.setattr(PatchPolicyModel, "from_config", Mock(side_effect=AssertionError("Model was rebuilt.")))

    policy.on_load_checkpoint(checkpoint)

    assert policy.model is model
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None


def test_checkpoint_refreshes_statistics_without_rebuilding_model() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert policy.config is not None
    stats = NormalizationParameters(mean=[10.0, 20.0], std=[2.0, 3.0], min=[10.0, 20.0], max=[14.0, 26.0])
    config = replace(
        policy.config,
        output_features=[replace(policy.config.output_features[0], normalization_data=stats)],
    )

    policy.on_load_checkpoint({"policy_config": config.to_dict()})

    assert policy.model is model
    assert policy.config == config
    assert policy.output_features == config.output_features
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    normalized = policy._preprocessor.normalize_actions(torch.tensor([[[14.0, 26.0]]]))
    torch.testing.assert_close(normalized, torch.ones_like(normalized))
    torch.testing.assert_close(policy._postprocessor(normalized), torch.tensor([[[14.0, 26.0]]]))


def test_checkpoint_rejects_incompatible_existing_model() -> None:
    policy = make_policy("diffusion")
    model = policy.model
    assert policy.config is not None
    config = policy.config

    with pytest.raises(ValueError, match="does not match"):
        policy.on_load_checkpoint({"policy_config": replace(config, hidden_dim=64).to_dict()})

    assert policy.model is model
    assert policy.config is config


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
def test_torch_export_round_trip(tmp_path: Path, action_head: str) -> None:
    """Torch exports preserve configuration, weights, and explicit-window inference."""
    original = make_policy(action_head, use_random_input_noise=False, image_size=(28, 42)).eval()

    original.export(tmp_path, backend=ExportBackend.TORCH)

    model_path = tmp_path / "patchpolicy.pt"
    assert model_path.is_file()
    manifest = Manifest.load(tmp_path)
    assert manifest.policy.source.class_path == "physicalai.policies.patch_policy.policy.PatchPolicy"
    assert manifest.model.artifacts["torch"] == model_path.name

    restored = PatchPolicy.load_from_checkpoint(model_path).eval()
    assert restored.config == original.config
    assert set(restored.state_dict()) == set(original.state_dict())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)

    adapter = TorchAdapter()
    adapter.load(model_path)
    for _ in range(3):
        images = torch.rand(2, 2, 3, 32, 32)
        observation = Observation(images={"front": images})
        torch.manual_seed(7)
        expected = original.predict_action_chunk(observation)
        torch.manual_seed(7)
        torch.testing.assert_close(restored.predict_action_chunk(observation), expected)
        torch.manual_seed(7)
        predicted = adapter.predict({IMAGES: {"front": images.numpy()}})
        np.testing.assert_allclose(predicted[ACTION], expected.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", [ExportBackend.ONNX, ExportBackend.OPENVINO, ExportBackend.EXECUTORCH])
def test_export_rejects_non_torch_backends(tmp_path: Path, backend: ExportBackend) -> None:
    """Patch Policy advertises and supports only Torch export."""
    policy = make_policy("diffusion")
    assert policy.get_supported_export_backends() == [ExportBackend.TORCH]
    with pytest.raises(NotImplementedError, match="not implemented"):
        policy.export(tmp_path, backend=backend)


def test_load_from_checkpoint_round_trip(tmp_path: Path) -> None:
    """Lightning checkpoint loading reconstructs modules before restoring their tensors."""
    original = make_policy("diffusion", use_random_input_noise=False)
    checkpoint = {
        "state_dict": original.state_dict(),
        "hyper_parameters": dict(original.hparams),
        "pytorch-lightning_version": L.__version__,
    }
    original.on_save_checkpoint(checkpoint)
    path = tmp_path / "patch_policy.ckpt"
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch  # Test-only checkpoint is generated from trusted local state.
    torch.save(checkpoint, path)

    restored = PatchPolicy.load_from_checkpoint(path)

    assert restored.config == original.config
    assert set(restored.state_dict()) == set(original.state_dict())
    assert restored.config.use_random_input_noise is False
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("initialization", ["features", "dataset", "checkpoint"])
def test_configure_model_initialization_paths(initialization: str, dummy_dataset: Any, tmp_path: Path) -> None:
    dataset = dummy_dataset(num_samples=2)
    dataset.delta_indices = {"action": [0, 1, 2, 3]}
    inputs = list(dataset.observation_features.values())
    outputs = list(dataset.action_features.values())
    kwargs: dict[str, Any] = {
        "encoder_name": "resnet18",
        "encoder_output": "average",
        "encoder_pretrained": False,
        "action_head": "diffusion",
        "n_obs_steps": 1,
        "chunk_size": 4,
        "n_action_steps": 2,
        "hidden_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "num_train_timesteps": 10,
        "num_inference_steps": 2,
    }
    policy = PatchPolicy(
        input_features=None if initialization == "dataset" else inputs,
        output_features=None if initialization == "dataset" else outputs,
        **kwargs,
    )
    if initialization == "checkpoint":
        checkpoint = {
            "state_dict": policy.state_dict(),
            "hyper_parameters": dict(policy.hparams),
            "pytorch-lightning_version": L.__version__,
        }
        policy.on_save_checkpoint(checkpoint)
        path = tmp_path / "fit.ckpt"
        torch.save(checkpoint, path)
        policy = PatchPolicy.load_from_checkpoint(path)
    model = policy.model
    datamodule = DataModule(dataset, train_batch_size=2, num_workers=0, persistent_workers=False)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(policy, datamodule=datamodule)

    assert trainer.global_step == 0
    assert policy.model is not None
    if model is not None:
        assert policy.model is model
    assert policy.config is not None
    assert policy.input_features == policy.config.input_features
    assert policy.output_features == policy.config.output_features
    assert policy._preprocessor is not None
    assert policy._postprocessor is not None
    assert policy._preprocessor.output_features == policy.config.output_features
    assert policy._postprocessor.output_features == policy.config.output_features
