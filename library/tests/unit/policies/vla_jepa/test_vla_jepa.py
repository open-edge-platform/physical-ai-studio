# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001, PLR2004

"""Native VLA-JEPA architecture, feature, and checkpoint contracts."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lightning
import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.policies import get_policy
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig, VLAJEPAModel
from physicalai.policies.vla_jepa.pretrained import _stats, resolve_pretrained  # noqa: PLC2701
from physicalai.policies.vla_jepa.processors import VLAJEPAPostprocessor, VLAJEPAPreprocessor


def test_config_round_trip(tiny_config: VLAJEPAConfig) -> None:
    """Serialization retains ordered features and embedded processor assets."""
    assert VLAJEPAConfig.from_dict(tiny_config.to_dict()) == tiny_config


@pytest.mark.parametrize(
    "settings",
    [
        {"chunk_size": 0},
        {"n_action_steps": 4},
        {"num_inference_steps": 0},
        {"action_dim": 4},
        {"state_dim": 3},
        {"state_normalization": "invalid"},
        {"pre_snap_gripper": True, "gripper_dim": 99},
        {"noise_beta_alpha": 0.0},
    ],
)
def test_invalid_configuration(tiny_config: VLAJEPAConfig, settings: dict) -> None:
    """Invalid model settings fail before allocating a backbone."""
    with pytest.raises(ValueError, match=r".+"):
        replace(tiny_config, **settings)


def test_factory_is_lazy_and_first_party() -> None:
    """Discovery does not require downloads or a LeRobot policy instance."""
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained") as resolve:
        policy = get_policy("vla_jepa")
    assert isinstance(policy, VLAJEPA)
    assert policy.model is None
    resolve.assert_not_called()


def test_native_loss_backward_and_denoising(tiny_config: VLAJEPAConfig, observation: Observation) -> None:
    """Gradients reach both stacks and inference returns the complete action horizon."""
    policy = VLAJEPA.from_config(tiny_config, gradient_checkpointing=True)
    loss, metrics = policy(observation)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert policy.model.qwen.language_model.embed_tokens.weight.grad is not None
    assert policy.model.action_model.action_encoder.layer1.weight.grad is not None
    assert metrics["loss"].requires_grad is False
    policy.eval()
    actions = policy.predict_action_chunk(observation)
    assert actions.shape == (2, 3, 3)
    assert torch.isfinite(actions).all()
    validation, _ = policy.compute_val_loss(observation)
    assert torch.isfinite(validation)


def test_padding_does_not_supervise(tiny_config: VLAJEPAConfig) -> None:
    """An entirely padded chunk contributes zero loss and remains differentiable."""
    model = VLAJEPAModel.from_config(tiny_config)
    context = {"tokens": torch.rand(2, 2, 32)}
    with patch.object(model, "_context", return_value=context):
        batch = {"action": torch.randn(2, 3, 3), "extra.action_is_pad": torch.ones(2, 3, dtype=torch.bool)}
        loss, _ = model.compute_loss(batch)
    assert loss.item() == 0
    loss.backward()


def test_denoising_uses_shared_euler_loop(tiny_config: VLAJEPAConfig) -> None:
    """Shared sampling agrees with the released Euler update."""
    head = VLAJEPAModel.from_config(tiny_config).action_model.eval()
    context = {"tokens": torch.rand(2, 2, 32), "state": torch.rand(2, 1, 2)}
    noise = torch.randn(2, 3, 3)
    expected = noise.clone()
    for step in range(2):
        times = torch.full((2,), step / 2)
        expected = 0.5 * head.denoise(expected, times, context) + expected
    torch.testing.assert_close(head.sample(context, noise=noise), expected)


def test_pre_norm_readout_matches_last_decoder_block(tiny_config: VLAJEPAConfig, observation: Observation) -> None:
    """The action head sees the trained pre-RMSNorm representation."""
    model = VLAJEPAModel.from_config(tiny_config).eval()
    batch = VLAJEPAPreprocessor(tiny_config)(observation.to_dict())
    captured = []
    handle = model.qwen.language_model.layers[-1].register_forward_hook(
        lambda _module, _inputs, outputs: captured.append(outputs[0] if isinstance(outputs, tuple) else outputs),
    )
    try:
        context = model._context(batch)
    finally:
        handle.remove()
    index = batch["embodied_positions"][:, :, None].expand(-1, -1, 32)
    torch.testing.assert_close(context["tokens"], captured[0].gather(1, index))


def test_configure_model_is_idempotent(tiny_config: VLAJEPAConfig) -> None:
    """Repeated Lightning construction hooks preserve weights and object identity."""
    policy = VLAJEPA.from_config(tiny_config)
    model = policy.model
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained") as resolve:
        policy.configure_model()
    assert policy.model is model
    resolve.assert_not_called()
    for name, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, weights[name])


def test_action_queue_and_reset(tiny_config: VLAJEPAConfig, observation: Observation) -> None:
    """The shared queue executes only the requested prefix and clears on reset."""
    policy = VLAJEPA.from_config(tiny_config).eval()
    chunk = torch.arange(18).reshape(2, 3, 3).float()
    with patch.object(policy, "predict_action_chunk", return_value=chunk) as predict:
        torch.testing.assert_close(policy.select_action(observation), chunk[:, 0])
        torch.testing.assert_close(policy.select_action(observation), chunk[:, 1])
        predict.assert_called_once()
        policy.reset()
        policy.select_action(observation)
        assert predict.call_count == 2


def test_feature_mismatch_fails_without_rebuilding(tiny_config: VLAJEPAConfig) -> None:
    """Changing a materialized feature contract cannot silently discard weights."""
    policy = VLAJEPA.from_config(tiny_config)
    dataset = MagicMock(spec=Dataset)
    dataset.observation_features = {"other_camera": replace(tiny_config.input_features[0], name="other_camera")}
    dataset.action_features = {"action": tiny_config.output_features[0]}
    policy._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=dataset))
    model = policy.model
    with pytest.raises(ValueError, match="features differ"):
        policy.setup("fit")
    assert policy.model is model


def test_lazy_training_resolves_dataset_before_materialization(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """Setup resolves features while module construction stays in configure_model."""
    dataset = MagicMock(spec=Dataset)
    dataset.observation_features = {str(feature.name): feature for feature in tiny_config.input_features}
    dataset.action_features = {str(feature.name): feature for feature in tiny_config.output_features}
    policy = VLAJEPA(enable_world_model=False)
    policy._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=dataset))
    weights = tmp_path / "model.safetensors"
    save_file(VLAJEPAModel.from_config(tiny_config).state_dict(), weights)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", return_value=(tiny_config, weights)):
        policy.setup("fit")
    assert policy.model is None
    assert policy.config.input_features == tiny_config.input_features
    policy.configure_model()
    assert isinstance(policy.model, VLAJEPAModel)


def test_changed_robot_projections_are_explicit_and_narrow(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """Embodiment adaptation cannot hide unrelated missing weights."""
    source = VLAJEPAModel.from_config(tiny_config)
    path = tmp_path / "model.safetensors"
    save_file(source.state_dict(), path)
    outputs = [Feature(name="action", ftype=FeatureType.ACTION, shape=(4,))]
    target = VLAJEPAModel.from_config(replace(tiny_config, output_features=outputs, action_dim=4))
    with pytest.raises(ValueError, match="shape mismatch"):
        target.load_weights(path)
    target.load_weights(path, reinitialize_projections=True)
    torch.testing.assert_close(
        target.qwen.language_model.embed_tokens.weight,
        source.qwen.language_model.embed_tokens.weight,
    )
    weights = source.state_dict()
    del weights["action_model.model.proj_out_2.weight"]
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing tensors"):
        target.load_weights(path, reinitialize_projections=True)


def test_converted_keys_ignore_only_training_only_components(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """World-model tensors are excluded while unknown tensors fail loading."""
    model = VLAJEPAModel.from_config(tiny_config)
    converted = {}
    for key, value in model.state_dict().items():
        renamed = "model.qwen.model.model." + key.removeprefix("qwen.") if key.startswith("qwen.") else "model." + key
        converted[renamed] = value
    converted["model.video_encoder.training_only"] = torch.ones(1)
    converted["model.video_predictor.training_only"] = torch.ones(1)
    path = tmp_path / "model.safetensors"
    save_file(converted, path)
    model.load_weights(path)
    converted["unknown.weight"] = torch.ones(1)
    save_file(converted, path)
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights(path)


def test_processors_round_trip_and_keep_dataset_gripper_units(tiny_config: VLAJEPAConfig) -> None:
    """Gripper conventions apply only when explicitly configured."""
    stats = NormalizationParameters(min=[0.0, -2.0, 0.0], max=[2.0, 2.0, 100.0])
    config = replace(tiny_config, output_features=[replace(tiny_config.output_features[0], normalization_data=stats)])
    post = VLAJEPAPostprocessor(config)
    actions = torch.tensor([[[-1.0, 0.0, 1.0]]])
    torch.testing.assert_close(post(actions), torch.tensor([[[0.0, 0.0, 100.0]]]))
    config = replace(config, pre_snap_gripper=True, binarize_gripper=True, gripper_dim=2)
    assert VLAJEPAPostprocessor(config)(actions)[0, 0, 2] == -1


def test_processor_rejects_missing_stats_and_injected_tokens(
    tiny_config: VLAJEPAConfig,
    observation: Observation,
) -> None:
    """Incomplete statistics and prompt token injection fail at the boundary."""
    outputs = [replace(tiny_config.output_features[0], normalization_data=None)]
    with pytest.raises(ValueError, match="statistics"):
        VLAJEPAPreprocessor(replace(tiny_config, output_features=outputs))
    observation.task = ["<|embodied_action|>", "move"]
    with pytest.raises(ValueError, match="invalid number"):
        VLAJEPAPreprocessor(tiny_config)(observation.to_dict())


def test_checkpoint_restores_offline(tiny_config: VLAJEPAConfig, observation: Observation, tmp_path: Path) -> None:
    """A trained Lightning checkpoint restores equivalent actions without downloads."""
    policy = VLAJEPA.from_config(tiny_config, gradient_checkpointing=False)
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    with patch.object(policy, "setup"):
        trainer.fit(policy, train_dataloaders=DataLoader([observation], batch_size=None))
    path = tmp_path / "policy.ckpt"
    trainer.save_checkpoint(path, weights_only=True)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network access")):
        restored = VLAJEPA.load_from_checkpoint(path, weights_only=True)
    assert restored.config == policy.config
    assert restored.pretrained_name_or_path is None
    policy.eval()
    restored.eval()
    torch.manual_seed(23)
    expected = policy.predict_action_chunk(observation)
    torch.manual_seed(23)
    torch.testing.assert_close(restored.predict_action_chunk(observation), expected)


@pytest.mark.parametrize(
    "filename",
    ["../weights.safetensors", "sub/weights.safetensors", "weights.pt", "..\\weights.safetensors"],
)
def test_remote_statistics_cannot_escape_checkpoint(tmp_path: Path, filename: str) -> None:
    """Untrusted asset names cannot traverse or select unsafe weight formats."""
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "normalizer_processor", "state_file": filename}]}),
    )
    with pytest.raises(ValueError, match="filename"):
        _stats(tmp_path)


def test_unpinned_custom_checkpoint_rejected_before_download() -> None:
    """Custom repositories must pin a reviewed commit before any network request."""
    with (
        patch("physicalai.policies.vla_jepa.pretrained.snapshot_download") as download,
        pytest.raises(ValueError, match="concrete"),
    ):
        resolve_pretrained("custom/model")
    download.assert_not_called()
