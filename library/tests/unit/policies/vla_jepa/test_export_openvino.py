# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001

"""Offline OpenVINO conversion and Runtime parity on a real tiny Qwen/DiT."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from physicalai.inference import InferenceModel
from physicalai.inference.manifest import Manifest

from physicalai.cli import export as export_cli
from physicalai.data import FeatureType, Observation
from physicalai.policies.mixins.peft import is_lora_injected
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig
from physicalai.policies.vla_jepa.export_openvino import VLAJEPAOpenVINOModel


@pytest.mark.parametrize("mode", ["plain", "identity", "lora", "dora", "joint_stateless", "no_state_feature"])
def test_openvino_runtime_parity(
    tiny_config: VLAJEPAConfig,
    jepa_config: VLAJEPAConfig,
    observation: Observation,
    tmp_path: Path,
    mode: str,
) -> None:
    """Vary task length and pixels after export; preserve normalized state and action units."""
    stats = tiny_config.output_features[0].normalization_data
    if mode != "identity":
        stats = replace(stats, min=[0.0, -2.0, 10.0], max=[2.0, 2.0, 20.0])
    base = jepa_config if mode == "joint_stateless" else tiny_config
    config = replace(
        base,
        output_features=[replace(base.output_features[0], normalization_data=stats)],
        lora_enabled=mode in {"lora", "dora"},
        lora_use_dora=mode == "dora",
        lora_rank=2,
        pre_snap_gripper=mode != "identity",
        binarize_gripper=mode != "identity",
        gripper_dim=2,
    )
    if mode == "joint_stateless":
        config = replace(
            config,
            input_features=[base.input_features[0], replace(base.input_features[0], name="wrist")],
            state_dim=0,
            n_action_steps=config.chunk_size,
        )
    if mode == "no_state_feature":
        # Released checkpoints retain a state encoder without declaring an observation state.
        config = replace(config, input_features=[config.input_features[0]])
    policy = VLAJEPA.from_config(config).eval()
    if config.use_lora:
        with torch.no_grad():
            for name, parameter in policy.model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.01)
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    original = policy.model
    policy.export(tmp_path, backend="openvino")
    assert policy.model is original
    assert is_lora_injected(policy.model) == config.use_lora
    assert not policy.training
    manifest = Manifest.load(tmp_path)
    assert manifest.model.artifacts == {"openvino": "vlajepa.xml"}
    assert (tmp_path / "vlajepa.xml").is_file()
    assert (tmp_path / "vlajepa.bin").is_file()
    assert (tmp_path / "tokenizer.json").is_file()
    # Runtime must not import the training library to run the export.
    for spec in [*manifest.model.preprocessors, *manifest.model.postprocessors]:
        assert not (spec.class_path or "").startswith("physicalai.policies")
    assert policy.config == config
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        runtime = InferenceModel(tmp_path, backend="openvino", device="CPU", INFERENCE_PRECISION_HINT="f32")
    for task in ("pick", "move the object to the other side of the table"):
        raw = {
            **{
                f"images.{feature.name}": torch.rand(1, *feature.shape)
                for feature in config.input_features
                if feature.ftype == FeatureType.VISUAL
            },
            "task": [task],
        }
        if any(feature.ftype == FeatureType.STATE for feature in config.input_features):
            raw["state"] = observation.state[:1]
        processed = policy._preprocessor(raw)
        noise = torch.randn(1, config.chunk_size, config.action_dim)
        wrapper = VLAJEPAOpenVINOModel(policy.model, config, processed["image_grid_thw"]).eval()
        processed["noise"] = noise
        with torch.no_grad():
            expected = policy._postprocessor(policy.model.predict_action_chunk(processed, noise=noise))
            # The graph normalizes raw state itself.
            graph_inputs = {**processed, "state": raw["state"]} if "state" in raw else processed
            torch.testing.assert_close(wrapper(graph_inputs), expected, rtol=1e-5, atol=1e-5)
        inputs = {name: value.numpy() if isinstance(value, torch.Tensor) else value for name, value in raw.items()}
        actual = runtime({**inputs, "task": np.array([task]), "noise": noise.numpy()})["action"]
        np.testing.assert_allclose(actual, expected[:, : config.n_action_steps].numpy(), rtol=1e-4, atol=1e-4)


def test_openvino_rejects_batched_trace(tiny_config: VLAJEPAConfig, observation: Observation, tmp_path: Path) -> None:
    """Do not silently export a graph that drops observations from a batch."""
    policy = VLAJEPA.from_config(tiny_config)
    with pytest.raises(ValueError, match="batch size one"):
        policy.to_openvino(tmp_path, policy._preprocessor(observation.to_dict()))


def test_openvino_cli_matches_api(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """CLI checkpoint loading produces the same Runtime contract as the Python API."""
    policy = VLAJEPA.from_config(tiny_config)
    policy.export(tmp_path / "source", backend="torch")
    policy.export(tmp_path / "api", backend="openvino")
    parser = export_cli.build_parser()
    args = parser.parse_args([
        "--policy",
        "physicalai.policies.VLAJEPA",
        "--ckpt_path",
        str(tmp_path / "source" / "vlajepa.pt"),
        "--backend",
        "openvino",
        "--output_dir",
        str(tmp_path / "cli"),
    ])
    with patch("physicalai.devices.get_available_device", return_value="cpu"):
        assert export_cli.run(parser, args) == 0
    assert Manifest.load(tmp_path / "api") == Manifest.load(tmp_path / "cli")
    assert policy.training
    assert policy.model.training


def test_openvino_failure_restores_live_model(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """Conversion failures leave adapters, weights, and training mode unchanged."""
    policy = VLAJEPA.from_config(replace(tiny_config, lora_enabled=True, lora_rank=2))
    original = policy.model
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    with (
        patch("openvino.convert_model", side_effect=RuntimeError("conversion failed")),
        pytest.raises(RuntimeError, match="conversion failed"),
    ):
        policy.export(tmp_path, backend="openvino")
    assert policy.model is original
    assert is_lora_injected(policy.model)
    assert policy.training
    assert policy.model.training
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
