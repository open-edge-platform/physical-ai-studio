# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upstream MolmoAct2 parameter storage and sensitive compute boundaries."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from lightning import LightningDataModule
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from physicalai.data import Observation
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


@pytest.mark.parametrize("autocast", [False, True])
def test_kv_projection_accepts_bf16_context_with_fp32_weights(
    tiny_molmoact2_config: MolmoAct2Config,
    autocast: bool,
) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)
    expert = policy.model.backbone.model.action_expert
    projection = expert.context_k_proj
    context = torch.randn(1, 3, projection.in_features, dtype=torch.bfloat16, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        projected = expert._project_kv(context, projection)

    assert projected.dtype == (torch.bfloat16 if autocast else torch.float32)
    assert torch.isfinite(projected).all()
    projected.float().square().mean().backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert projection.weight.grad.dtype == torch.float32
    assert torch.isfinite(projection.weight.grad).all()


def test_direct_prediction_after_bf16_mixed_fit(
    tiny_molmoact2_config: MolmoAct2Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        tiny_molmoact2_config,
        flow_matching_num_steps=1,
        num_flow_timesteps=1,
        image_start_token_id=64,
        image_end_token_id=65,
        image_patch_id=66,
        image_placeholder_token_id=67,
        low_res_image_start_token_id=64,
        image_use_col_tokens=False,
        image_processor_size={"height": 28, "width": 28},
    )
    policy = MolmoAct2.from_config(config)
    # Keep real observation/image processing, but avoid downloading a tokenizer.
    monkeypatch.setattr(
        policy._preprocessor._tokenizers,
        "tokenize_prompts",
        lambda _: (torch.tensor([[1, 67, 2]]), torch.ones(1, 3, dtype=torch.long)),
    )
    monkeypatch.setattr(type(policy._preprocessor._tokenizers), "pad_token_id", property(lambda _: 0))
    observation = Observation(
        images={"image": torch.rand(1, 3, 28, 28)},
        state=torch.zeros(1, 4),
        action=torch.zeros(1, config.chunk_size, 4),
        task=["pick up the object"],
    )
    datamodule = LightningDataModule()
    datamodule.train_dataset = Mock(spec=Dataset)
    datamodule.train_dataset.observation_features = {feature.name: feature for feature in config.input_features}
    datamodule.train_dataset.action_features = {feature.name: feature for feature in config.output_features}
    monkeypatch.setattr(datamodule, "train_dataloader", lambda: DataLoader([observation], batch_size=None))
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(policy, datamodule=datamodule)

    assert trainer.global_step == 1
    assert policy.model.backbone.lm_head.weight.dtype == torch.bfloat16
    assert policy.model.backbone.model.action_expert.context_k_proj.weight.dtype == torch.float32
    assert not torch.is_autocast_enabled("cpu")
    actions = policy.eval().predict_action_chunk(observation)

    assert actions.shape == (1, config.chunk_size, 4)
    assert actions.dtype == torch.float32
    assert torch.isfinite(actions).all()


def test_mixed_storage_kv_projection_can_be_traced(tiny_molmoact2_config: MolmoAct2Config) -> None:
    policy = MolmoAct2.from_config(tiny_molmoact2_config)

    class KVProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert = policy.model.backbone.model.action_expert

        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return self.expert._project_kv(context, self.expert.context_k_proj)

    module = KVProjection().eval()
    context = torch.randn(1, 3, module.expert.context_k_proj.in_features, dtype=torch.bfloat16)
    traced = torch.jit.trace(module, context)
    actual = traced(context)

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, module(context))
