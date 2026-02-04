# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pi0 policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads).
"""

from __future__ import annotations

import pytest
import torch
from getiaction.config import Config
from getiaction.policies.pi0 import Pi0, Pi05, Pi0Config


class TestPi0Config:
    """Tests for Pi0Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Pi0Config()
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_300m"
        assert config.variant == "pi0"
        assert config.dtype == "bfloat16"
        assert config.tune_action_expert is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Pi0Config(
            variant="pi05",
            chunk_size=100,
            n_action_steps=50,
            learning_rate=1e-4,
            tune_vision_encoder=True,
            paligemma_variant="gemma_2b",
        )
        assert config.variant == "pi05"
        assert config.chunk_size == 100
        assert config.n_action_steps == 50
        assert config.learning_rate == 1e-4
        assert config.tune_vision_encoder is True
        assert config.paligemma_variant == "gemma_2b"

    def test_training_config_values(self) -> None:
        """Test training-related configuration values."""
        config = Pi0Config()
        assert config.learning_rate == 2.5e-5
        assert config.weight_decay == 1e-10
        assert config.warmup_steps == 1000
        assert config.decay_steps == 30000
        assert config.decay_lr == 2.5e-6
        assert config.grad_clip_norm == 1.0

    def test_flow_matching_config_values(self) -> None:
        """Test flow matching configuration values."""
        config = Pi0Config()
        assert config.time_beta_alpha == 1.5
        assert config.time_beta_beta == 1.0
        assert config.time_scale == 0.999
        assert config.time_offset == 0.001
        assert config.time_min_period == 4e-3
        assert config.time_max_period == 4.0
        assert config.num_inference_steps == 10

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="chunk_size"):
            Pi0Config(chunk_size=50, n_action_steps=100)

    def test_variant_validation(self) -> None:
        """Test variant must be pi0 or pi05."""
        with pytest.raises(ValueError, match="variant"):
            Pi0Config(variant="invalid")  # type: ignore[arg-type]

    def test_paligemma_variant_validation(self) -> None:
        """Test paligemma_variant must be gemma_2b."""
        with pytest.raises(ValueError, match="paligemma_variant"):
            Pi0Config(paligemma_variant="gemma_300m")
        with pytest.raises(ValueError, match="paligemma_variant"):
            Pi0Config(paligemma_variant="invalid")

    def test_action_expert_variant_validation(self) -> None:
        """Test action_expert_variant must be valid."""
        with pytest.raises(ValueError, match="action_expert_variant"):
            Pi0Config(action_expert_variant="invalid")

    def test_is_pi05_property(self) -> None:
        """Test is_pi05 property."""
        config_pi0 = Pi0Config(variant="pi0")
        config_pi05 = Pi0Config(variant="pi05")
        assert config_pi0.is_pi05 is False
        assert config_pi05.is_pi05 is True

    def test_use_lora_property(self) -> None:
        """Test use_lora property."""
        config_no_lora = Pi0Config(lora_rank=0)
        config_with_lora = Pi0Config(lora_rank=8)
        assert config_no_lora.use_lora is False
        assert config_with_lora.use_lora is True

    def test_inheritance_and_serialization(self) -> None:
        """Test config inherits from base Config and supports serialization."""
        config = Pi0Config(chunk_size=100, learning_rate=1e-4)
        assert isinstance(config, Config)

        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 100
        assert config_dict["learning_rate"] == 1e-4

        restored = Pi0Config.from_dict(config_dict)
        assert restored.chunk_size == 100
        assert restored.learning_rate == 1e-4

    def test_max_token_len_auto_computed(self) -> None:
        """Test max_token_len is auto-computed based on variant."""
        config_pi0 = Pi0Config(variant="pi0", max_token_len=None)
        config_pi05 = Pi0Config(variant="pi05", max_token_len=None)
        assert config_pi0.max_token_len == 48
        assert config_pi05.max_token_len == 200


class TestPi0Policy:
    """Tests for Pi0 Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test lazy initialization doesn't create model."""
        policy = Pi0()
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        policy = Pi0(
            chunk_size=100,
            learning_rate=1e-4,
            tune_vision_encoder=True,
        )
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.learning_rate == 1e-4
        assert policy.hparams.tune_vision_encoder is True
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 100

    def test_config_attribute(self) -> None:
        """Test Pi0 policy has config attribute."""
        policy = Pi0(chunk_size=100, learning_rate=1e-4)

        assert policy.config is not None
        assert policy.config.chunk_size == 100
        assert policy.config.learning_rate == 1e-4

    def test_n_action_steps(self) -> None:
        """Test n_action_steps is correctly set."""
        policy = Pi0(n_action_steps=25, chunk_size=50)
        assert policy._n_action_steps == 25
        assert policy.config.n_action_steps == 25

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        """Test methods raise ValueError if model not initialized."""
        from getiaction.data import Observation

        policy = Pi0()
        dummy_obs = Observation(state=torch.randn(1, 10))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(dummy_obs)


class TestPi05Policy:
    """Tests for Pi05 (Pi0.5) policy alias."""

    def test_pi05_creates_pi05_variant(self) -> None:
        """Test Pi05 creates policy with variant='pi05'."""
        policy = Pi05()
        assert policy.config.variant == "pi05"
        assert policy.config.is_pi05 is True

    def test_pi05_inherits_from_pi0(self) -> None:
        """Test Pi05 inherits from Pi0."""
        policy = Pi05()
        assert isinstance(policy, Pi0)

    def test_pi05_with_custom_args(self) -> None:
        """Test Pi05 accepts custom arguments."""
        policy = Pi05(chunk_size=100, learning_rate=1e-4)
        assert policy.config.variant == "pi05"
        assert policy.config.chunk_size == 100
        assert policy.config.learning_rate == 1e-4


class TestPi0Preprocessor:
    """Tests for Pi0 preprocessor functions."""

    def test_make_pi0_preprocessors(self) -> None:
        """Test make_pi0_preprocessors returns callables."""
        from getiaction.policies.pi0.preprocessor import make_pi0_preprocessors

        preprocessor, postprocessor = make_pi0_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            chunk_size=50,
            stats=None,
            image_resolution=(224, 224),
            max_token_len=48,
        )
        assert callable(preprocessor)
        assert callable(postprocessor)

    def test_preprocessor_is_nn_module(self) -> None:
        """Test that preprocessors are nn.Module instances."""
        from getiaction.policies.pi0.preprocessor import (
            Pi0Postprocessor,
            Pi0Preprocessor,
        )
        from torch import nn

        preprocessor = Pi0Preprocessor()
        postprocessor = Pi0Postprocessor(action_dim=7)

        assert isinstance(preprocessor, nn.Module)
        assert isinstance(postprocessor, nn.Module)

    def test_preprocessor_default_values(self) -> None:
        """Test preprocessor default configuration values."""
        from getiaction.policies.pi0.preprocessor import Pi0Preprocessor

        preprocessor = Pi0Preprocessor()

        assert preprocessor.max_state_dim == 32
        assert preprocessor.max_action_dim == 32
        assert preprocessor.image_resolution == (224, 224)
        assert preprocessor.max_token_len == 48

    def test_preprocessor_custom_values(self) -> None:
        """Test preprocessor with custom configuration values."""
        from getiaction.policies.pi0.preprocessor import Pi0Preprocessor

        preprocessor = Pi0Preprocessor(
            max_state_dim=64,
            max_action_dim=16,
            image_resolution=(512, 512),
            max_token_len=64,
        )

        assert preprocessor.max_state_dim == 64
        assert preprocessor.max_action_dim == 16
        assert preprocessor.image_resolution == (512, 512)
        assert preprocessor.max_token_len == 64


class TestGetPolicy:
    """Tests for get_policy with Pi0."""

    def test_get_pi0_policy(self) -> None:
        """Test creating Pi0 policy via get_policy."""
        from getiaction.policies import get_policy

        policy = get_policy("pi0", source="getiaction")
        assert policy.__class__.__name__ == "Pi0"

    def test_get_pi05_policy(self) -> None:
        """Test creating Pi05 policy via get_policy."""
        from getiaction.policies import get_policy

        policy = get_policy("pi05", source="getiaction")
        assert policy.__class__.__name__ == "Pi05"
        assert policy.config.variant == "pi05"

    def test_case_insensitive(self) -> None:
        """Test policy name is case-insensitive."""
        from getiaction.policies import get_policy

        policy = get_policy("PI0")
        assert policy.__class__.__name__ == "Pi0"
