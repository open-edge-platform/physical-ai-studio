# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001

"""Boundary and lifecycle regressions for native VLA-JEPA."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from physicalai.data import NormalizationParameters, Observation
from physicalai.data.dataset import Dataset
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig, VLAJEPAModel
from physicalai.policies.vla_jepa.pretrained import _stats  # noqa: PLC2701
from physicalai.policies.vla_jepa.processors import VLAJEPAPostprocessor, VLAJEPAPreprocessor


def test_hub_statistics_symlinks_are_supported(tmp_path: Path) -> None:
    """A validated Hub asset may link from a snapshot into its sibling blob cache."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob.safetensors"
    save_file({"action.min": torch.zeros(3), "action.max": torch.ones(3)}, blob)
    (snapshot / "stats.safetensors").symlink_to(blob)
    (snapshot / "policy_preprocessor.json").write_text(
        json.dumps({
            "steps": [{"registry_name": "normalizer_processor", "state_file": "stats.safetensors"}],
        }),
    )
    assert _stats(snapshot)["action"]["max"] == [1.0] * 3


def test_partial_state_weights_are_not_an_embodiment_change(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """Projection adaptation must not hide a truncated state encoder."""
    model = VLAJEPAModel.from_config(tiny_config)
    weights = model.state_dict()
    del weights["action_model.state_encoder.layer2.weight"]
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing tensors"):
        model.load_weights(path, reinitialize_projections=True)


def test_failed_materialization_can_be_retried(tiny_config: VLAJEPAConfig) -> None:
    """Invalid processor metadata cannot leave a half-initialized model installed."""
    policy = VLAJEPA(pretrained_name_or_path=None)
    policy.config = replace(
        tiny_config,
        output_features=[replace(tiny_config.output_features[0], normalization_data=None)],
    )
    with pytest.raises(ValueError, match="statistics"):
        policy.configure_model()
    assert policy.model is None
    policy.config = tiny_config
    policy.configure_model()
    assert isinstance(policy.model, VLAJEPAModel)


def test_trainer_resume_defers_pretrained_resolution(tiny_config: VLAJEPAConfig) -> None:
    """Lightning resume obtains architecture from its checkpoint, not the Hub."""
    dataset = MagicMock(spec=Dataset)
    dataset.observation_features = {str(feature.name): feature for feature in tiny_config.input_features}
    dataset.action_features = {str(feature.name): feature for feature in tiny_config.output_features}
    policy = VLAJEPA()
    policy._trainer = SimpleNamespace(datamodule=SimpleNamespace(train_dataset=dataset), ckpt_path="resume.ckpt")
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained") as resolve:
        policy.setup("fit")
        policy.configure_model()
        assert policy.model is None
        policy.on_load_checkpoint({"policy_config": tiny_config.to_dict()})
    resolve.assert_not_called()
    assert isinstance(policy.model, VLAJEPAModel)


def test_head_only_training_keeps_backbone_frozen(tiny_config: VLAJEPAConfig) -> None:
    """Mode changes cannot reactivate a frozen Qwen backbone."""
    policy = VLAJEPA.from_config(tiny_config, train_action_head_only=True)
    policy.train()
    assert not policy.model.qwen.training
    assert not any(parameter.requires_grad for parameter in policy.model.qwen.parameters())
    assert all(parameter.requires_grad for parameter in policy.model.action_model.parameters())


@pytest.mark.parametrize("field", ["state", "action"])
def test_nonfinite_vectors_are_rejected(tiny_config: VLAJEPAConfig, observation: Observation, field: str) -> None:
    """Invalid robot vectors fail before producing a non-finite training loss."""
    getattr(observation, field).fill_(torch.nan)
    with pytest.raises(ValueError, match=r"State tensor|Actions must"):
        VLAJEPAPreprocessor(tiny_config)(observation.to_dict())


def test_invalid_normalization_bounds_are_rejected(tiny_config: VLAJEPAConfig) -> None:
    """Malformed statistics cannot reverse the action scale."""
    bad = NormalizationParameters(min=[1.0] * 3, max=[0.0] * 3)
    config = replace(tiny_config, output_features=[replace(tiny_config.output_features[0], normalization_data=bad)])
    with pytest.raises(ValueError, match="bounds"):
        VLAJEPAPreprocessor(config)


@pytest.mark.parametrize("mode", ["MEAN_STD", "IDENTITY"])
def test_unbounded_actions_are_not_clipped(tiny_config: VLAJEPAConfig, mode: str) -> None:
    """Clipping to [-1, 1] is valid only for bounded normalization modes."""
    stats = NormalizationParameters(mean=[1.0] * 3, std=[2.0] * 3)
    config = replace(
        tiny_config,
        action_normalization=mode,
        output_features=[replace(tiny_config.output_features[0], normalization_data=stats)],
    )
    actions = torch.tensor([[[-2.0, 0.0, 2.0]]])
    expected = actions * 2 + 1 if mode == "MEAN_STD" else actions
    torch.testing.assert_close(VLAJEPAPostprocessor(config)(actions), expected)


def test_clipping_preserves_unnormalized_action_dimensions(tiny_config: VLAJEPAConfig) -> None:
    """Feature masks also exempt raw action dimensions from normalized clipping."""
    stats = NormalizationParameters(min=[-1.0] * 3, max=[1.0] * 3, mask=[True, False, True])
    config = replace(tiny_config, output_features=[replace(tiny_config.output_features[0], normalization_data=stats)])
    actions = torch.tensor([[[-2.0, 2.0, 2.0]]])
    torch.testing.assert_close(VLAJEPAPostprocessor(config)(actions), torch.tensor([[[-1.0, 2.0, 1.0]]]))


def test_single_camera_tensor_matches_named_camera(tiny_config: VLAJEPAConfig, observation: Observation) -> None:
    """Dataset batches may represent one camera as a tensor rather than a dictionary."""
    processor = VLAJEPAPreprocessor(tiny_config)
    expected = processor(observation.to_dict())
    observation.images = observation.images["front"]
    actual = processor(observation.to_dict())
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def test_config_version_is_checked(tiny_config: VLAJEPAConfig) -> None:
    """Future config formats fail explicitly rather than changing reconstruction."""
    with pytest.raises(ValueError, match="version"):
        replace(tiny_config, format_version=3)
