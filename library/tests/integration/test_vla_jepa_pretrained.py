# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001

"""Released safetensors loading and CUDA backward for native VLA-JEPA."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from physicalai.data import FeatureType, Observation
from physicalai.inference import InferenceModel
from physicalai.policies import VLAJEPA, VLAJEPAConfig
from physicalai.policies.mixins.peft import is_lora_injected
from physicalai.policies.vla_jepa.processors import VLAJEPAPreprocessor


@pytest.mark.slow
@pytest.mark.requires_download
@pytest.mark.parametrize("enable_world_model", [False, True])
@pytest.mark.parametrize("lora_enabled", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU and the released multi-GB checkpoint")
def test_pretrained_inference_backward_and_offline_processors(
    *,
    enable_world_model: bool,
    lora_enabled: bool,
) -> None:
    """Real checkpoint tensors support inference, gradients, and self-contained processor restore."""
    policy = (
        VLAJEPA
        .from_pretrained(
            gradient_checkpointing=True,
            enable_world_model=enable_world_model,
            lora_enabled=lora_enabled,
        )
        .to("cuda")
        .eval()
    )
    config = policy.config
    observation = Observation(
        images={
            str(feature.name): torch.rand(
                (1, config.num_video_frames, *feature.shape) if enable_world_model else (1, *feature.shape),
                device="cuda",
            )
            for feature in config.input_features
            if feature.ftype == FeatureType.VISUAL
        },
        task=["pick up the object"],
    )
    restored_config = VLAJEPAConfig.from_dict(config.to_dict())
    assert restored_config == config
    with patch(
        "physicalai.policies.vla_jepa.pretrained.AutoProcessor.from_pretrained",
        side_effect=AssertionError("download"),
    ):
        restored_preprocessor = VLAJEPAPreprocessor(restored_config).to("cuda")
    expected = policy._preprocessor(observation.to_dict())
    actual = restored_preprocessor(observation.to_dict())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    actions = policy.predict_action_chunk(observation)
    assert actions.shape == (1, config.chunk_size, config.action_dim)
    assert actions.isfinite().all()
    observation.action = torch.rand_like(actions) * 2 - 1
    policy.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(observation)
    loss.backward()
    assert loss.isfinite()
    if lora_enabled:
        assert is_lora_injected(policy.model)
        for name, parameter in policy.model.named_parameters():
            assert parameter.requires_grad == ("lora_" in name)
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all(), name
        assert policy.model.qwen.language_model.layers[0].self_attn.q_proj.lora_B["default"].weight.grad is not None
        assert policy.model.action_model.action_encoder.layer1.lora_B["default"].weight.grad is not None
    else:
        assert policy.model.qwen.language_model.embed_tokens.weight.grad is not None
        assert policy.model.action_model.action_encoder.layer1.weight.grad is not None
    if enable_world_model:
        projection = policy.model.video_predictor.predictor_embed
        gradient = projection.lora_B["default"].weight.grad if lora_enabled else projection.weight.grad
        assert gradient is not None
        assert all(parameter.grad is None for parameter in policy.model.video_encoder.parameters())
        assert not policy.model.video_encoder.training
    else:
        assert policy.model.video_encoder is None
        assert policy.model.video_predictor is None


@pytest.mark.slow
@pytest.mark.requires_download
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU and the released multi-GB checkpoint")
def test_pretrained_lora_torch_export(tmp_path: Path) -> None:
    """Released bfloat16 weights export and reload offline through the Torch Runtime adapter."""
    policy = VLAJEPA.from_pretrained(lora_enabled=True, lora_dropout=0.0).to("cuda").eval()
    config = policy.config
    observation = Observation(
        images={
            str(feature.name): torch.rand(1, *feature.shape, device="cuda")
            for feature in config.input_features
            if feature.ftype == FeatureType.VISUAL
        },
        task=["pick up the object"],
    )
    # Nonzero updates exercise merging rather than exporting unchanged base weights.
    with torch.no_grad():
        for name, parameter in policy.model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=1e-4)
    torch.manual_seed(31)
    expected = policy.predict_action_chunk(observation)
    policy.export(tmp_path, backend="torch")
    assert is_lora_injected(policy.model)
    inputs = {f"images.{name}": value.cpu().numpy() for name, value in observation.images.items()}
    inputs["task"] = observation.task
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        runtime = InferenceModel(tmp_path, backend="torch", device="cuda")
    assert not is_lora_injected(runtime.adapter._policy.model)
    torch.manual_seed(31)
    actual = torch.from_numpy(runtime(inputs)["action"]).to(expected)
    # Folding float32 adapters into bfloat16 base layers rounds their updates.
    torch.testing.assert_close(actual, expected[:, : config.n_action_steps], atol=2e-2, rtol=2e-2)


@pytest.mark.slow
@pytest.mark.requires_download
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU and the released multi-GB checkpoint")
def test_pretrained_openvino_export(tmp_path: Path) -> None:
    """Released bfloat16 weights serialize safely and run offline on the OpenVINO CPU adapter."""
    torch.manual_seed(41)
    policy = VLAJEPA.from_pretrained().to("cuda").eval()
    config = policy.config
    policy.export(tmp_path, backend="openvino")
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        runtime = InferenceModel(tmp_path, backend="openvino", device="CPU", INFERENCE_PRECISION_HINT="f32")
    for task in ("pick up the object", "move the object to the other side of the table"):
        raw = {
            f"images.{feature.name}": torch.rand(1, *feature.shape, device="cuda")
            for feature in config.input_features if feature.ftype == FeatureType.VISUAL
        }
        raw["task"] = [task]
        noise = torch.randn(1, config.chunk_size, config.action_dim, device="cuda")
        processed = policy._preprocessor(raw)
        with torch.no_grad():
            expected = policy._postprocessor(policy.model.predict_action_chunk(processed, noise=noise))
        inputs = {name: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for name, value in raw.items()}
        actual = torch.from_numpy(runtime({**inputs, "noise": noise.cpu().numpy()})["action"]).to(expected)
        # CPU FP32 kernels and CUDA bfloat16 kernels round differently across the full sampler.
        torch.testing.assert_close(actual, expected[:, :config.n_action_steps], atol=3e-2, rtol=3e-2)
