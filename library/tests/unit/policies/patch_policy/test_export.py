# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export contract tests for Patch Policy."""

# ruff: noqa: S101

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from physicalai.inference.data import InferenceFeatureType
from physicalai.inference.model import InferenceModel

from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.export import ExportBackend
from physicalai.policies.patch_policy import PatchPolicy
from physicalai.policies.patch_policy.export import action_normalization_stats, image_normalization_stats

if TYPE_CHECKING:
    from pathlib import Path


def _features() -> tuple[list[Feature], list[Feature]]:
    image_stats = NormalizationParameters(mean=[0.2, 0.4, 0.6], std=[0.1, 0.2, 0.3])
    return (
        [
            Feature(name="front", ftype=FeatureType.VISUAL, shape=(3, 28, 28), normalization_data=image_stats),
            Feature(name="wrist", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        ],
        [
            Feature(
                name="action.arm",
                ftype=FeatureType.ACTION,
                shape=(2,),
                normalization_data=NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.5]),
            ),
            Feature(name="action.gripper", ftype=FeatureType.ACTION, shape=(1,)),
        ],
    )


def _policy(action_head: str = "behavior_transformer", *, n_obs_steps: int = 1) -> PatchPolicy:
    inputs, outputs = _features()
    return PatchPolicy(
        input_features=inputs,
        output_features=outputs,
        encoder_name="dinov2",
        encoder_output="patches",
        encoder_pretrained=False,
        image_size=(28, 28),
        action_head=action_head,  # type: ignore[arg-type]
        n_obs_steps=n_obs_steps,
        chunk_size=4,
        n_action_steps=2,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        num_inference_steps=2,
        eta=0.0,
        use_random_input_noise=False,
    )


def test_supported_backends_and_schemas() -> None:
    """The initial contract exposes ordered raw cameras and a full action chunk."""
    policy = _policy()

    assert policy.get_supported_export_backends() == [ExportBackend.TORCH, ExportBackend.OPENVINO]
    assert policy.inputs_schema is not None
    assert [feature.name for feature in policy.inputs_schema] == ["images.front", "images.wrist"]
    assert all(feature.ftype is InferenceFeatureType.VISUAL for feature in policy.inputs_schema)
    assert policy.outputs_schema is not None
    assert policy.outputs_schema[0].shape == (4, 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoder_name", "resnet18", "dinov2"),
        ("encoder_output", "average", "patches"),
    ],
)
def test_rejects_configs_outside_initial_contract(field: str, value: object, message: str) -> None:
    """Unsupported temporal and encoder configurations fail before conversion."""
    policy = PatchPolicy.from_config(replace(_policy().config, **{field: value}))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        _ = policy.extra_export_args


def test_flat_and_nested_camera_inputs_match() -> None:
    """Inherited flat samples and normal nested observations pack identically."""
    policy = _policy()
    assert policy._preprocessor is not None  # noqa: SLF001
    front = torch.rand(1, 3, 28, 28)
    wrist = torch.rand(1, 3, 28, 28)

    nested = policy._preprocessor({"images": {"front": front, "wrist": wrist}})  # noqa: SLF001
    flat = policy._preprocessor({"images.front": front, "images.wrist": wrist})  # noqa: SLF001

    torch.testing.assert_close(flat["images"], nested["images"])
    assert flat["images"].shape == (1, 1, 2, 3, 28, 28)


def test_temporal_input_schema_and_preprocessing() -> None:
    """Multi-frame exports expose and preserve the configured temporal window."""
    policy = _policy(n_obs_steps=3)
    assert policy.inputs_schema is not None
    assert policy.inputs_schema[0].shape == (3, 3, 28, 28)
    assert policy._preprocessor is not None  # noqa: SLF001
    front = torch.rand(1, 3, 3, 28, 28)
    wrist = torch.rand(1, 3, 3, 28, 28)

    processed = policy._preprocessor({"images.front": front, "images.wrist": wrist})  # noqa: SLF001

    assert processed["images"].shape == (1, 3, 2, 3, 28, 28)


def test_temporal_export_rejects_wrong_frame_count() -> None:
    """Export preprocessing applies the same strict frame count as eager PyTorch."""
    policy = _policy(n_obs_steps=2)
    assert policy._preprocessor is not None  # noqa: SLF001

    with pytest.raises(ValueError, match="Expected 2 observation timestep"):
        policy._preprocessor(  # noqa: SLF001
            {
                "images.front": torch.rand(1, 3, 28, 28),
                "images.wrist": torch.rand(1, 3, 28, 28),
            },
        )


def test_runtime_stats_match_stacked_feature_contract() -> None:
    """Missing feature statistics are represented by false identity masks."""
    inputs, outputs = _features()

    image_stats = image_normalization_stats(inputs)
    action_stats = action_normalization_stats(outputs)

    assert image_stats is not None
    assert action_stats is not None
    assert torch.tensor(image_stats["images"]["mean"]).shape == (2, 3, 1, 1)
    assert not torch.tensor(image_stats["images"]["mask"])[1].any()
    assert action_stats["action"]["mask"] == [True, True, False]


def test_export_specs_use_shared_runtime_processors() -> None:
    """OpenVINO uses shared normalization and postprocessing around camera packing."""
    args = _policy().extra_export_args
    openvino = args["openvino"]
    torch_args = args["torch"]

    assert [spec.type for spec in openvino.preprocessors_specs] == ["patch_policy", "normalize", "to_float_tensor"]
    assert [spec.type for spec in openvino.postprocessors_specs] == ["denormalize", "action_chunk_trimmer"]
    assert [spec.type for spec in torch_args.preprocessors_specs] == ["to_float_tensor"]
    assert torch_args.postprocessors_specs == []


def test_torch_export_writes_checkpoint_and_manifest(tmp_path: Path) -> None:
    """The public export API produces a reloadable-style Torch package."""
    policy = _policy()

    policy.export(tmp_path, backend="torch")

    assert (tmp_path / "patchpolicy.pt").is_file()
    assert (tmp_path / "manifest.json").is_file()
    restored = PatchPolicy.load_from_checkpoint(tmp_path / "patchpolicy.pt")
    assert restored.config == policy.config

    with InferenceModel(tmp_path, device="cpu") as runtime:
        image = np.ones((1, 3, 28, 28), dtype=np.float32)
        outputs = runtime({"images": {"front": image, "wrist": image}})
    assert outputs["action"].shape == (1, 2, 3)


@pytest.mark.parametrize("action_head", ["behavior_transformer", "diffusion"])
def test_openvino_export_writes_model_package(tmp_path: Path, action_head: str) -> None:
    """Both initial action heads convert through the public OpenVINO API."""
    policy = _policy(action_head)

    policy.export(tmp_path, backend="openvino")

    assert (tmp_path / "patchpolicy.xml").is_file()
    assert (tmp_path / "patchpolicy.bin").is_file()
    assert (tmp_path / "manifest.json").is_file()


def test_openvino_export_supports_temporal_observation_window(tmp_path: Path) -> None:
    """Behavior Transformer conversion preserves multiple observation frames."""
    policy = _policy(n_obs_steps=2)

    policy.export(tmp_path, backend="openvino")

    assert (tmp_path / "patchpolicy.xml").is_file()


def test_torch_export_supports_temporal_observation_window(tmp_path: Path) -> None:
    """Reloaded Torch policies consume the same explicit temporal camera window."""
    policy = _policy(n_obs_steps=2)
    policy.export(tmp_path, backend="torch")
    images = np.ones((1, 2, 3, 28, 28), dtype=np.float32)

    with InferenceModel(tmp_path, device="cpu") as runtime:
        outputs = runtime({"images": {"front": images, "wrist": images}})

    assert outputs["action"].shape == (1, 2, 3)
