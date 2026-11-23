# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upstream MolmoAct2 parameter storage and sensitive compute boundaries."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from physicalai.data.dataset import Dataset
from physicalai.policies.molmoact2 import MolmoAct2, MolmoAct2Config
from physicalai.policies.molmoact2.components.rms_norm import RMSNorm
from physicalai.policies.molmoact2.components.text import MolmoAct2RotaryEmbedding, apply_rotary_pos_emb
from physicalai.policies.molmoact2.optimizer import MolmoAct2AdamW


@pytest.mark.parametrize("affine", [False, True])
def test_rms_norm_matches_fp32_affine_then_downcast(affine: bool) -> None:
    norm = RMSNorm(16, elementwise_affine=affine)
    if norm.weight is not None:
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.7, 1.3, 16))
    x = torch.linspace(-2, 3, 32).reshape(2, 16).bfloat16().requires_grad_()
    expected = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps)
    if norm.weight is not None:
        expected = expected * norm.weight

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = norm(x)

    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    actual.float().sum().backward()
    assert x.grad is not None
    if norm.weight is not None:
        assert norm.weight.grad.dtype == torch.float32


def test_text_rope_frequencies_ignore_autocast() -> None:
    rope = MolmoAct2RotaryEmbedding(head_dim=16, rope_theta=5_000_000.0)
    positions = torch.tensor([[257, 513, 1025]])
    x = torch.ones(1, 3, 16, dtype=torch.bfloat16)
    expected = rope(x, positions)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = rope(x, positions)

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_text_rope_rotation_uses_fp32_intermediates() -> None:
    q = torch.linspace(-2, 3, 32).reshape(1, 2, 1, 16).bfloat16()
    k = q.flip(-1)
    cos = torch.linspace(0.1, 0.9, 16).reshape(1, 1, 16).bfloat16()
    sin = 1 - cos
    expected = apply_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float())

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = apply_rotary_pos_emb(q, k, cos, sin)

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_fit_storage_preserves_sensitive_values_and_freezes_only_base_embedding(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
    eager: bool,
    precision: str,
) -> None:
    config = tiny_molmoact2_config
    policy = MolmoAct2.from_config(config) if eager else MolmoAct2(pretrained_name_or_path=None)
    if not eager:
        monkeypatch.setattr(policy, "initialize_model", lambda: policy._initialize_from_config(config))
    trainer = Mock()
    trainer.precision = precision
    trainer.datamodule.train_dataset = Mock(spec=Dataset)
    policy._trainer = trainer
    monkeypatch.setattr(policy, "_dataset_features", lambda _: (config.input_features, config.output_features))
    policy.setup("fit")

    model = policy._require_model()
    backbone = model.backbone.model
    assert not backbone.transformer.wte.embedding.requires_grad
    assert backbone.transformer.wte.new_embedding.requires_grad
    assert all(p.dtype == torch.float32 for p in backbone.action_expert.parameters())
    expected_dtype = torch.bfloat16 if precision == "bf16-mixed" else torch.float32
    assert model.backbone.lm_head.weight.dtype == expected_dtype
    assert backbone.vision_backbone.image_projector.w1.weight.dtype == expected_dtype

    # Applying preparation again must not round retained FP32 values through BF16.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm) and module.weight is not None:
                module.weight.fill_(1.000123)
        next(backbone.action_expert.parameters()).fill_(1.000123)
    retained = {name: p.clone() for name, p in model.named_parameters() if p.dtype == torch.float32}
    policy._prepare_training_parameters()
    for name, parameter in model.named_parameters():
        if name in retained:
            torch.testing.assert_close(parameter, retained[name], rtol=0, atol=0)
    for module in model.modules():
        if isinstance(module, MolmoAct2RotaryEmbedding):
            assert module.inv_freq.dtype == torch.float32


def test_fit_preserves_lora_adapter_dtype(tiny_molmoact2_config: MolmoAct2Config) -> None:
    policy = MolmoAct2.from_config(replace(tiny_molmoact2_config, lora_enabled=True))
    policy._trainer = Mock(precision="bf16-mixed")
    policy._prepare_training_parameters()

    adapters = [p for name, p in policy.model.named_parameters() if ".lora_" in name]
    assert adapters
    assert all(p.dtype == torch.float32 for p in adapters)


def test_vocab_freeze_can_be_disabled(tiny_molmoact2_config: MolmoAct2Config) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config, freeze_vocab=False)
    policy._trainer = Mock(precision="32-true")
    policy._prepare_training_parameters()

    assert policy.model.backbone.model.transformer.wte.embedding.requires_grad


def test_action_expert_uses_bf16_compute_with_fp32_gradients_and_moments(
    tiny_molmoact2_config: MolmoAct2Config,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    policy._trainer = Mock(precision="bf16-mixed")
    policy._prepare_training_parameters()
    expert = policy.model.backbone.model.action_expert
    projection = next(module for module in expert.modules() if isinstance(module, torch.nn.Linear))
    optimizer = MolmoAct2AdamW(expert.parameters(), lr=5e-5, group_grad_clip_norm=1.0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = projection(torch.ones(2, projection.in_features))
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert projection.weight.grad.dtype == torch.float32
    optimizer.step()
    assert optimizer.state[projection.weight]["exp_avg"].dtype == torch.float32
    assert optimizer.state[projection.weight]["exp_avg_sq"].dtype == torch.float32
