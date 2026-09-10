# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MolmoAct2 unit tests."""

from pathlib import Path

import pytest

from physicalai.data.observation import Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2 import MolmoAct2Config


@pytest.fixture
def molmoact2_features() -> tuple[list[Feature], list[Feature]]:
    stats = NormalizationParameters(q01=[-1.0] * 4, q99=[1.0] * 4)
    inputs = [
        Feature(name="image", ftype=FeatureType.VISUAL, shape=(3, 28, 28)),
        Feature(name="state", ftype=FeatureType.STATE, shape=(4,), normalization_data=stats),
    ]
    outputs = [Feature(name="action", ftype=FeatureType.ACTION, shape=(4,), normalization_data=stats)]
    return inputs, outputs


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tiny_molmoact2_config(
    molmoact2_features: tuple[list[Feature], list[Feature]],
    tokenizer_dir: Path,
) -> MolmoAct2Config:
    inputs, outputs = molmoact2_features
    return MolmoAct2Config(
        input_features=inputs,
        output_features=outputs,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        additional_vocab_size=4,
        num_hidden_layers=1,
        intermediate_size=64,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_hidden_layers=1,
        vision_num_attention_heads=2,
        vision_num_key_value_heads=2,
        vision_head_dim=16,
        image_default_input_size=(28, 28),
        image_patch_size=14,
        image_num_pos=4,
        adapter_vit_layers=(-1,),
        adapter_hidden_size=32,
        adapter_num_attention_heads=2,
        adapter_num_key_value_heads=2,
        adapter_head_dim=16,
        adapter_intermediate_size=64,
        adapter_text_hidden_size=32,
        action_expert_max_action_dim=4,
        action_expert_hidden_size=32,
        action_expert_num_layers=1,
        action_expert_num_heads=2,
        action_expert_ffn_multiple_of=16,
        action_expert_timestep_embed_dim=16,
        max_action_dim=4,
        chunk_size=4,
        n_action_steps=2,
        tokenizer_name_or_path=str(tokenizer_dir),
    )
