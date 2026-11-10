# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy export contracts and offline Runtime parity."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physicalai import policies
from physicalai.data import Feature, FeatureType, NormalizationParameters, Observation
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.inference import InferenceModel
from physicalai.inference.component_factory import instantiate_component, resolve_artifact
from physicalai.inference.manifest import ComponentSpec, Manifest
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.policies import patch_policy
from physicalai.policies.patch_policy import policy as policy_module
from physicalai.policies.patch_policy.config import PatchPolicyConfig
from physicalai.policies.patch_policy.export import PatchPolicyExportMixin
from physicalai.policies.patch_policy.model import PatchPolicyModel
from physicalai.policies.patch_policy.policy import PatchPolicy
from physicalai.policies.patch_policy.processors import make_policy_processors


def test_export_mixin_supports_only_torch() -> None:
    assert issubclass(PatchPolicyExportMixin, ExportablePolicyMixin)
    assert PatchPolicyExportMixin.get_supported_export_backends() == [ExportBackend.TORCH]


def test_policy_declares_torch_support_without_building_model() -> None:
    policy = PatchPolicy(encoder_pretrained=False, hidden_dim=4, num_layers=1, num_heads=1)
    assert isinstance(policy, PatchPolicyExportMixin)
    assert policy.model is None
    assert policy.get_supported_export_backends() == [ExportBackend.TORCH]


def test_backend_list_is_not_shared_mutable_state() -> None:
    backends = PatchPolicyExportMixin.get_supported_export_backends()
    backends.append("onnx")
    assert PatchPolicyExportMixin.get_supported_export_backends() == [ExportBackend.TORCH]
    assert PatchPolicy.get_supported_export_backends() == [ExportBackend.TORCH]


def _run_processors(
    specs: list[ComponentSpec],
    values: dict[str, np.ndarray],
    export_dir: Path,
    base: type,
) -> dict[str, np.ndarray]:
    for spec in specs:
        component = instantiate_component(base, resolve_artifact(spec, export_dir))
        values = component(values)
    return values


@pytest.mark.parametrize("action_head", ["diffusion", "behavior_transformer"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(("batch", "timesteps", "source_size"), [(1, 1, (9, 11)), (2, 3, (4, 5))])
def test_model_only_processors_match_torch(
    tiny_encoder, small_config, tmp_path, action_head, dtype, batch, timesteps, source_size,
) -> None:
    image_size = (5, 7)
    config = replace(
        small_config,
        action_head=action_head,
        n_obs_steps=timesteps,
        action_dim=4,
        input_features=[
            Feature(
                name="observation.images.wrist",
                ftype=FeatureType.VISUAL,
                shape=(3, *source_size),
                normalization_data=NormalizationParameters(mean=[0.1, 0.3, 0.5], std=[0.2, 0.4, 0.8]),
            ),
            Feature(name="images.front", ftype=FeatureType.VISUAL, shape=(3, *source_size)),
        ],
        output_features=[
            Feature(
                name="arm",
                ftype=FeatureType.ACTION,
                shape=(2,),
                normalization_data=NormalizationParameters(min=[-2.0, 3.0], max=[6.0, 9.0], mask=[True, False]),
            ),
            Feature(
                name="gripper",
                ftype=FeatureType.ACTION,
                shape=(1,),
                normalization_data=NormalizationParameters(min=[-5.0]),
            ),
            Feature(name="other", ftype=FeatureType.ACTION, shape=(1,)),
        ],
    )
    class Exporter(PatchPolicyExportMixin):
        @property
        def config(self):
            return config

    exporter = Exporter()
    exporter.model = SimpleNamespace(encoder=tiny_encoder("tiny", "patch", image_size=image_size))
    parameters = exporter.extra_export_args["openvino"]
    preprocessors = [ComponentSpec.model_validate(spec.model_dump(mode="json")) for spec in parameters.preprocessors_specs]
    postprocessors = [ComponentSpec.model_validate(spec.model_dump(mode="json")) for spec in parameters.postprocessors_specs]
    torch_preprocessor, torch_postprocessor = make_policy_processors(config, image_size)
    rng = np.random.default_rng(42)
    images = {
        name: rng.random((batch, timesteps, 3, *source_size), dtype=np.float32)
        for name in ("front", "wrist")
    }
    if dtype == np.uint8:
        images = {name: (image * 255).astype(dtype) for name, image in images.items()}
    actual = _run_processors(preprocessors, {"images": images}, tmp_path, Preprocessor)
    expected = torch_preprocessor({"images": {name: torch.from_numpy(image) for name, image in images.items()}})

    assert list(actual) == list(expected) == ["images.wrist", "images.front"]
    for key in expected:
        assert actual[key].shape == (batch, timesteps, 3, *image_size)
        assert actual[key].dtype == np.float32
        np.testing.assert_allclose(actual[key], expected[key].numpy(), rtol=1e-5, atol=2e-6)

    actions = rng.uniform(-1, 1, (batch, config.chunk_size, config.action_dim)).astype(np.float32)
    actual_actions = _run_processors(postprocessors, {"action": actions.copy()}, tmp_path, Postprocessor)["action"]
    expected_actions = torch_postprocessor(torch.from_numpy(actions)).numpy()
    assert actual_actions.shape == (batch, config.n_action_steps, config.action_dim)
    np.testing.assert_allclose(actual_actions, expected_actions, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual_actions[..., 1:], actions[:, : config.n_action_steps, 1:])
    if action_head == "behavior_transformer":
        np.testing.assert_array_equal(actual_actions, actions[:, : config.n_action_steps])


def test_torch_export_round_trip_is_offline(tiny_encoder, small_config, tmp_path, monkeypatch) -> None:
    torch.manual_seed(42)
    policy = PatchPolicy.from_config(small_config).cpu().eval()
    images = np.random.default_rng(42).integers(0, 256, (2, small_config.n_obs_steps, 3, 7, 9), dtype=np.uint8)
    with torch.inference_mode():
        expected = policy.predict_action_chunk(Observation(images={"front": torch.from_numpy(images)})).numpy()

    export_dir = tmp_path / "export"
    policy.to_torch(export_dir)
    manifest = Manifest.load(export_dir)
    assert manifest.policy.name == "patchpolicy"
    assert manifest.model.artifacts == {"torch": "patchpolicy.pt"}
    assert (export_dir / manifest.model.artifacts["torch"]).is_file()
    assert manifest.model.preprocessors[0].flat_params["n_obs_steps"] == small_config.n_obs_steps
    assert not manifest.model.postprocessors
    assert not (export_dir / "stats.safetensors").exists()
    assert manifest.model.input_features[0].init_args["shape"] == [small_config.n_obs_steps, 3, 4, 4]
    assert manifest.model.output_features[0].init_args["shape"] == [small_config.n_action_steps, small_config.action_dim]

    def forbid_download(*args, **kwargs):
        raise AssertionError("Torch export restoration must not download encoder artifacts")

    monkeypatch.setattr(policy_module, "resolve_encoder_artifacts", forbid_download)
    restored = InferenceModel(export_dir, backend="torch", device="cpu")
    actual = restored({"images.front": images})["action"]
    assert actual.shape == expected.shape == (2, small_config.n_action_steps, small_config.action_dim)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(restored({"images.front": images})["action"], actual)


def test_unconfigured_torch_export_fails(tmp_path) -> None:
    policy = PatchPolicy(encoder_pretrained=False)
    export_dir = tmp_path / "unconfigured"
    with pytest.raises(RuntimeError, match="config is not initialized"):
        policy.to_torch(export_dir)


def test_package_exports_policy_config_and_model_classes() -> None:
    assert patch_policy.__all__ == ["PatchPolicy", "PatchPolicyConfig", "PatchPolicyModel"]
    assert patch_policy.PatchPolicy is PatchPolicy
    assert patch_policy.PatchPolicyConfig is PatchPolicyConfig
    assert patch_policy.PatchPolicyModel is PatchPolicyModel
    assert policies.PatchPolicy is PatchPolicy
    assert policies.PatchPolicyConfig is PatchPolicyConfig
    assert policies.PatchPolicyModel is PatchPolicyModel
    assert policies.get_physicalai_policy_class("patch_policy") is PatchPolicy
