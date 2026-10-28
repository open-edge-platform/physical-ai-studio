# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: S101, SLF001

"""LoRA training, offline restore, and Torch Runtime export contracts."""

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning
import numpy as np
import pytest
import torch
from physicalai.inference.manifest import Manifest
from torch import nn
from torch.utils.data import DataLoader

from physicalai.cli import export as export_cli
from physicalai.data import Observation
from physicalai.export import ExportBackend
from physicalai.inference import InferenceModel
from physicalai.policies.mixins.peft import is_lora_injected
from physicalai.policies.vla_jepa import VLAJEPA, VLAJEPAConfig, VLAJEPAModel


@pytest.mark.parametrize("use_dora", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_adapters_train_and_restore(
    tiny_config: VLAJEPAConfig,
    jepa_config: VLAJEPAConfig,
    observation: Observation,
    video_observation: Observation,
    tmp_path: Path,
    *,
    use_dora: bool,
    joint: bool,
) -> None:
    """Only adapters train, including the world predictor, and checkpoints resume offline."""
    config = replace(jepa_config if joint else tiny_config, lora_enabled=True, lora_rank=2, lora_use_dora=use_dora)
    policy = VLAJEPA.from_config(config, gradient_checkpointing=True)
    batch = video_observation if joint else observation
    trainer = lightning.Trainer(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    with patch.object(policy, "setup"):
        trainer.fit(policy, train_dataloaders=DataLoader([batch], batch_size=None))
    assert is_lora_injected(policy.model)
    for name, parameter in policy.model.named_parameters():
        assert parameter.requires_grad == ("lora_" in name)
        if "lora_B" in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    if joint:
        assert not is_lora_injected(policy.model.video_encoder)
        assert is_lora_injected(policy.model.video_predictor)
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path, weights_only=True)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        restored = VLAJEPA.load_from_checkpoint(path, weights_only=True)
    assert restored.config == config
    assert is_lora_injected(restored.model)
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    policy.eval()
    restored.eval()
    torch.manual_seed(17)
    expected = policy.predict_action_chunk(batch)
    torch.manual_seed(17)
    torch.testing.assert_close(restored.predict_action_chunk(batch), expected)


def test_default_targets_cover_active_linears_only(jepa_config: VLAJEPAConfig) -> None:
    """Default targets include each active head/predictor Linear, but no frozen encoders."""
    model = VLAJEPAModel.from_config(jepa_config)
    targets = model.get_default_peft_targets()
    matched = {name for name, module in model.named_modules() if re.fullmatch(targets, name)}
    expected = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith(("qwen.language_model.", "action_model.", "video_predictor."))
        and module.weight.requires_grad
    }
    assert matched == expected


@pytest.mark.parametrize("use_lora", [False, True])
def test_lora_optimizer_scales_both_learning_rates(tiny_config: VLAJEPAConfig, *, use_lora: bool) -> None:
    """Adapter LR scaling preserves the backbone/head ratio and covers all trainable weights."""
    config = replace(tiny_config, lora_enabled=use_lora, lora_rank=2, lora_lr_scale=4)
    policy = VLAJEPA.from_config(config, scheduler_warmup_steps=0)
    policy._trainer = SimpleNamespace(estimated_stepping_batches=10)
    optimizer = policy.configure_optimizers()["optimizer"]
    scale = 4 if use_lora else 1
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([1e-5 * scale, 1e-4 * scale])
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in policy.model.parameters() if p.requires_grad
    }


@pytest.mark.parametrize("use_lora,use_dora", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("joint", [False, True])
def test_torch_export_loads_in_runtime(
    tiny_config: VLAJEPAConfig,
    jepa_config: VLAJEPAConfig,
    observation: Observation,
    tmp_path: Path,
    *,
    use_lora: bool,
    use_dora: bool,
    joint: bool,
) -> None:
    """Merged Torch artifacts load offline in Runtime with matching denormalized actions."""
    base = jepa_config if joint else tiny_config
    # Non-identity statistics expose missing or double action denormalization.
    stats = replace(base.output_features[0].normalization_data, min=[0.0, -2.0, 10.0], max=[2.0, 2.0, 20.0])
    config = replace(
        base,
        lora_enabled=use_lora,
        lora_use_dora=use_dora,
        lora_rank=2,
        output_features=[replace(base.output_features[0], normalization_data=stats)],
    )
    policy = VLAJEPA.from_config(config).eval()
    if use_lora:
        with torch.no_grad():
            for name, parameter in policy.model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.01)
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    torch.manual_seed(19)
    expected = policy.predict_action_chunk(observation)
    policy.export(tmp_path, backend="torch")
    cli_path = tmp_path / "cli"
    parser = export_cli.build_parser()
    args = parser.parse_args([
        "--policy",
        "physicalai.policies.VLAJEPA",
        "--ckpt_path",
        str(tmp_path / "vlajepa.pt"),
        "--backend",
        "torch",
        "--output_dir",
        str(cli_path),
    ])
    with patch("physicalai.devices.get_available_device", return_value="cpu"):
        assert export_cli.run(parser, args) == 0
    assert Manifest.load(cli_path) == Manifest.load(tmp_path)
    assert policy.config == config
    assert is_lora_injected(policy.model) == use_lora
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    checkpoint = torch.load(tmp_path / "vlajepa.pt", weights_only=True)
    assert checkpoint["policy_config"]["lora_enabled"] is False
    assert not any("lora_" in name for name in checkpoint["state_dict"])
    manifest = Manifest.load(tmp_path)
    assert manifest.model.artifacts == {"torch": "vlajepa.pt"}
    assert policy.get_supported_export_backends() == [ExportBackend.TORCH, ExportBackend.OPENVINO]
    assert [feature.name for feature in policy.inputs_schema] == ["images.front", "state", "task"]
    assert policy.outputs_schema[0].shape == (3, 3)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", side_effect=AssertionError("network")):
        runtime = InferenceModel(tmp_path, backend="torch", device="cpu")
    inputs = {
        "images.front": observation.images["front"].numpy(),
        "state": observation.state.numpy(),
        "task": observation.task,
    }
    torch.manual_seed(19)
    actual = runtime(inputs)["action"]
    np.testing.assert_allclose(actual, expected[:, :2].numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("settings", [{"lora_rank": 0}, {"lora_dropout": 1.0}, {"lora_lr_scale": 0}])
def test_lora_config_validation(tiny_config: VLAJEPAConfig, settings: dict) -> None:
    """The policy config runs shared PEFT validation."""
    with pytest.raises(ValueError, match="lora_"):
        replace(tiny_config, lora_enabled=True, **settings)


def test_lora_custom_targets_and_conflicting_freeze(tiny_config: VLAJEPAConfig) -> None:
    """Custom targets restrict training without an incompatible backbone freeze flag."""
    config = replace(tiny_config, lora_enabled=True, lora_rank=2, lora_target_modules=("action_decoder.layer2",))
    policy = VLAJEPA.from_config(config)
    assert not is_lora_injected(policy.model.qwen)
    assert is_lora_injected(policy.model.action_model)
    with pytest.raises(ValueError, match="incompatible"):
        VLAJEPA.from_config(config, train_action_head_only=True)
    with pytest.raises(ValueError, match="incompatible"):
        VLAJEPA(lora_enabled=True, train_action_head_only=True)


@pytest.mark.parametrize("adapter_dtype", ["float32", "auto"])
def test_lora_adapter_precision(tiny_config: VLAJEPAConfig, observation: Observation, adapter_dtype: str) -> None:
    """A low-precision backbone supports float32 or inherited-precision adapters."""
    config = replace(tiny_config, dtype="bfloat16", lora_enabled=True, lora_rank=2, lora_adapter_dtype=adapter_dtype)
    policy = VLAJEPA.from_config(config)
    dtype = torch.float32 if adapter_dtype == "float32" else torch.bfloat16
    assert policy.model.qwen.language_model.layers[0].self_attn.q_proj.lora_A["default"].weight.dtype == dtype
    loss, _ = policy(observation)
    loss.backward()
    assert torch.isfinite(loss)


def test_failed_lora_injection_can_be_retried(tiny_config: VLAJEPAConfig) -> None:
    """Invalid targets cannot leave a frozen, half-initialized policy installed."""
    policy = VLAJEPA(pretrained_name_or_path=None)
    policy.config = replace(tiny_config, lora_enabled=True, lora_target_modules=("missing_projection",))
    with pytest.raises(RuntimeError, match="zero target"):
        policy.configure_model()
    assert policy.model is None
    policy.config = replace(tiny_config, lora_enabled=True, lora_rank=2)
    policy.configure_model()
    assert is_lora_injected(policy.model)


@pytest.mark.parametrize("backend", ["torch", "openvino"])
def test_uninitialized_export_is_rejected(tmp_path: Path, backend: str) -> None:
    """A lazy policy fails explicitly instead of writing an unusable artifact."""
    with pytest.raises(RuntimeError, match="uninitialized"):
        VLAJEPA().export(tmp_path, backend=backend)


def test_lazy_constructor_preserves_lora_settings(tiny_config: VLAJEPAConfig) -> None:
    """Dataset/pretrained resolution carries CLI adapter options into the resolved config."""
    rank = 4
    policy = VLAJEPA(enable_world_model=False, lora_enabled=True, lora_rank=rank, lora_use_dora=True)
    with patch("physicalai.policies.vla_jepa.policy.resolve_pretrained", return_value=(tiny_config, None)):
        policy._resolve_configuration()
    policy.configure_model()
    assert policy.config.lora_enabled
    assert policy.config.lora_rank == rank
    assert policy.config.lora_use_dora
    assert is_lora_injected(policy.model)


def test_export_failure_restores_adapters(tiny_config: VLAJEPAConfig, tmp_path: Path) -> None:
    """Export errors cannot strip adapters or alter live training weights."""
    policy = VLAJEPA.from_config(replace(tiny_config, lora_enabled=True, lora_rank=2))
    before = {name: tensor.clone() for name, tensor in policy.state_dict().items()}
    with (
        patch.object(policy, "create_manifest", side_effect=RuntimeError("write failed")),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        policy.export(tmp_path, backend="torch")
    assert is_lora_injected(policy.model)
    assert policy.training
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
