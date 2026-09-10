# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MolmoAct2 configuration."""

import pytest

from physicalai.config import Config
from physicalai.data import Feature, FeatureType, NormalizationParameters
from physicalai.policies.molmoact2 import MolmoAct2Config


def test_defaults_match_pretrained_architecture() -> None:
    config = MolmoAct2Config()

    assert (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) == (2560, 36, 32)
    assert (config.chunk_size, config.n_action_steps, config.max_action_dim) == (30, 30, 32)
    assert config.tokenizer_name_or_path == "allenai/MolmoAct2"


def test_custom_fields() -> None:
    config = MolmoAct2Config(chunk_size=8, n_action_steps=4, hidden_size=128)

    assert (config.chunk_size, config.n_action_steps, config.hidden_size) == (8, 4, 128)


def test_serialization_round_trip() -> None:
    config = MolmoAct2Config(chunk_size=8, n_action_steps=4, tokenizer_config={"pad_token": ""})
    restored = MolmoAct2Config.from_dict(config.to_dict())

    assert isinstance(restored, Config)
    assert (restored.chunk_size, restored.n_action_steps) == (8, 4)
    assert restored.tokenizer_config == {"pad_token": ""}


def test_nested_normalization_serialization_round_trip() -> None:
    normalization = NormalizationParameters(
        mean=[[0.1], [0.2], [0.3]],
        std=[[[0.4]], [[0.5]], [[0.6]]],
    )
    feature = Feature(
        name="camera",
        ftype=FeatureType.VISUAL,
        shape=(3, 8, 8),
        normalization_data=normalization,
    )
    config = MolmoAct2Config(input_features=[feature])

    restored = MolmoAct2Config.from_dict(config.to_dict())

    assert restored == config


def test_policy_runtime_options_are_not_model_config() -> None:
    data = MolmoAct2Config().to_dict()

    assert {
        "compile_model",
        "gradient_checkpointing",
        "openvino_compress_to_fp16",
        "optimizer_lr",
        "use_lora",
    }.isdisjoint(data)


def test_rollout_settings_validation() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        MolmoAct2Config(chunk_size=0)
    with pytest.raises(ValueError, match="n_action_steps"):
        MolmoAct2Config(n_action_steps=0)
    with pytest.raises(ValueError, match="cannot be greater"):
        MolmoAct2Config(chunk_size=2, n_action_steps=3)
    with pytest.raises(ValueError, match="n_obs_steps"):
        MolmoAct2Config(n_obs_steps=0)
    with pytest.raises(ValueError, match="max_action_dim"):
        MolmoAct2Config(max_action_dim=0)
