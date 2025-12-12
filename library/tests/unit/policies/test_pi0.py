# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pi0/Pi0.5 policy."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from getiaction.policies.pi0 import Pi0, Pi0Config
from getiaction.policies.pi0.components.attention import (
    AdaRMSNorm,
    make_attention_mask_2d,
    prepare_4d_attention_mask,
)
from getiaction.policies.pi0.model import Pi0Model
from getiaction.policies.pi0.preprocessor import (
    NormStats,
    Pi0Postprocessor,
    Pi0Preprocessor,
    make_pi0_preprocessors,
)


# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestPi0Config:
    """Tests for Pi0Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Pi0Config()
        assert config.variant == "pi0"
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_300m"
        assert not config.is_pi05
        assert not config.use_lora

    @pytest.mark.parametrize("variant,expected_token_len", [("pi0", 48), ("pi05", 200)])
    def test_variant_token_length(self, variant: str, expected_token_len: int) -> None:
        """Test variant auto-sets token length."""
        config = Pi0Config(variant=variant)  # type: ignore[arg-type]
        assert config.max_token_len == expected_token_len
        assert config.is_pi05 == (variant == "pi05")

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Pi0Config(
            variant="pi0",
            max_token_len=100,
            action_dim=14,
            action_horizon=100,
            lora_rank=16,
            lora_alpha=32,
        )
        assert config.max_token_len == 100
        assert config.use_lora
        assert config.lora_rank == 16

    def test_lora_disabled_when_rank_zero(self) -> None:
        """Test LoRA disabled when rank is 0."""
        assert not Pi0Config(lora_rank=0).use_lora

    def test_invalid_variants_raise_error(self) -> None:
        """Test invalid variant parameters raise ValueError."""
        with pytest.raises(ValueError):
            Pi0Config(variant="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Pi0Config(paligemma_variant="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Pi0Config(action_expert_variant="invalid")


# ============================================================================ #
# Component Tests (Attention, Preprocessor)                                    #
# ============================================================================ #


class TestAdaRMSNorm:
    """Tests for AdaRMSNorm layer."""

    def test_forward_with_and_without_conditioning(self) -> None:
        """Test forward pass with and without conditioning."""
        norm = AdaRMSNorm(hidden_size=128)
        x = torch.randn(2, 10, 128)
        cond = torch.randn(2, 128)

        output_no_cond = norm(x)
        output_with_cond = norm(x, conditioning=cond)

        assert output_no_cond.shape == x.shape
        assert output_with_cond.shape == x.shape
        assert not torch.allclose(output_no_cond, output_with_cond)


class TestAttentionMasks:
    """Tests for attention mask construction."""

    def test_bidirectional_and_causal_masks(self) -> None:
        """Test bidirectional and causal attention patterns."""
        pad_masks = torch.ones(2, 5, dtype=torch.bool)

        # Bidirectional (all zeros in att_masks)
        mask_2d = make_attention_mask_2d(pad_masks, torch.zeros(2, 5))
        assert mask_2d.shape == (2, 5, 5)
        assert mask_2d.all()

        # Causal (all ones in att_masks)
        mask_2d = make_attention_mask_2d(pad_masks, torch.ones(2, 5))
        expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
        assert torch.equal(mask_2d[0], expected)

    def test_prefix_lm_and_padding_masks(self) -> None:
        """Test prefix-LM pattern and padding handling."""
        # Prefix-LM: first 3 bidirectional, last 3 causal
        pad_masks = torch.ones(1, 6, dtype=torch.bool)
        att_masks = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.float)
        mask_2d = make_attention_mask_2d(pad_masks, att_masks)
        assert mask_2d[0, 0, :3].all()  # Token 0 attends to prefix
        assert not mask_2d[0, 3, 4]  # Token 3 can't attend to future

        # Padding mask
        pad_masks = torch.tensor([[True, True, True, False, False]], dtype=torch.bool)
        mask_2d = make_attention_mask_2d(pad_masks, torch.zeros(1, 5))
        assert mask_2d[0, :3, :3].all()  # Valid tokens attend
        assert not mask_2d[0, :3, 3:].any()  # Padded positions masked

    def test_4d_mask_preparation(self) -> None:
        """Test 4D attention mask preparation."""
        mask_2d = torch.tensor([[[True, False], [True, True]]], dtype=torch.bool)
        mask_4d = prepare_4d_attention_mask(mask_2d, dtype=torch.float16)

        assert mask_4d.shape == (1, 1, 2, 2)
        assert mask_4d.dtype == torch.float16
        assert mask_4d[0, 0, 0, 0] == 0.0  # Allowed
        assert torch.isinf(mask_4d[0, 0, 0, 1]) and mask_4d[0, 0, 0, 1] < 0  # Masked (negative inf)


class TestPreprocessor:
    """Tests for Pi0Preprocessor."""

    def test_state_padding_and_truncation(self) -> None:
        """Test state padding and truncation."""
        # Padding
        preprocessor = Pi0Preprocessor(max_state_dim=32)
        state = torch.randn(2, 14)
        processed = preprocessor._process_state(state)
        assert processed.shape == (2, 32)
        assert torch.allclose(processed[:, :14], state)

        # Truncation
        preprocessor = Pi0Preprocessor(max_state_dim=10)
        state = torch.randn(2, 20)
        processed = preprocessor._process_state(state)
        assert processed.shape == (2, 10)
        assert torch.allclose(processed, state[:, :10])

    def test_action_processing(self) -> None:
        """Test action processing handles different input shapes."""
        preprocessor = Pi0Preprocessor(max_action_dim=32, action_horizon=50)

        # 3D input
        actions = torch.randn(2, 30, 14)
        processed = preprocessor._process_actions(actions)
        assert processed.shape == (2, 50, 32)

        # 2D input (expanded)
        actions = torch.randn(2, 14)
        processed = preprocessor._process_actions(actions)
        assert processed.shape == (2, 50, 32)

    @pytest.mark.parametrize("use_quantile", [True, False])
    def test_normalization(self, use_quantile: bool) -> None:
        """Test quantile and z-score normalization."""
        if use_quantile:
            stats = NormStats(q01=np.array([0.0, 0.0]), q99=np.array([1.0, 1.0]))
            state = torch.tensor([[0.5, 0.5]])  # Mid-value
        else:
            stats = NormStats(mean=np.array([0.0, 1.0]), std=np.array([1.0, 2.0]))
            state = torch.tensor([[0.0, 1.0]])  # At mean

        preprocessor = Pi0Preprocessor(
            max_state_dim=2,
            use_quantile_norm=use_quantile,
            stats={"state": stats},
        )
        processed = preprocessor._normalize(state, stats)
        assert torch.allclose(processed, torch.zeros(1, 2), atol=1e-5)


class TestPostprocessor:
    """Tests for Pi0Postprocessor."""

    def test_action_truncation_and_denormalization(self) -> None:
        """Test action truncation and quantile denormalization."""
        stats = NormStats(q01=np.array([0.0]), q99=np.array([1.0]))
        postprocessor = Pi0Postprocessor(
            action_dim=14,
            max_action_dim=32,
            use_quantile_norm=True,
            stats={"actions": stats},
        )

        # Truncation
        outputs = {"actions": torch.randn(2, 50, 32)}
        processed = postprocessor(outputs)
        assert processed["actions"].shape == (2, 50, 14)

        # Denormalization
        outputs = {"actions": torch.tensor([[[-1.0]], [[1.0]]])}
        processed = postprocessor(outputs)
        assert torch.allclose(processed["actions"][0], torch.tensor([[0.0]]), atol=1e-5)
        assert torch.allclose(processed["actions"][1], torch.tensor([[1.0]]), atol=1e-5)


class TestPreprocessorFactory:
    """Tests for preprocessor factory function."""

    def test_create_preprocessors(self) -> None:
        """Test creates matching preprocessor/postprocessor pair."""
        # Without stats
        preprocessor, postprocessor = make_pi0_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            action_horizon=50,
            env_action_dim=14,
        )
        assert isinstance(preprocessor, Pi0Preprocessor)
        assert isinstance(postprocessor, Pi0Postprocessor)
        assert postprocessor.action_dim == 14

        # With stats
        stats = {"state": {"mean": [0.0], "std": [1.0]}}
        preprocessor, postprocessor = make_pi0_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            stats=stats,
        )
        assert preprocessor.stats is not None


# ============================================================================ #
# Policy and Model Tests                                                       #
# ============================================================================ #


@pytest.fixture
def mock_paligemma_load():
    """Mock PaliGemmaWithExpert to skip HuggingFace model loading.

    This mocks _ensure_loaded and set_trainable_parameters to avoid downloading
    large models in unit tests. Integration tests should use real models.
    """
    with patch(
        "getiaction.policies.pi0.components.gemma.PaliGemmaWithExpert._ensure_loaded",
        return_value=None,
    ), patch(
        "getiaction.policies.pi0.components.gemma.PaliGemmaWithExpert.set_trainable_parameters",
        return_value=None,
    ):
        yield


@pytest.mark.usefixtures("mock_paligemma_load")
class TestPi0Policy:
    """Tests for Pi0Policy and Pi0Model."""

    @pytest.fixture
    def config(self) -> Pi0Config:
        """Create a minimal config for testing."""
        return Pi0Config(
            paligemma_variant="gemma_300m",
            action_expert_variant="gemma_300m",
            action_dim=7,
            action_horizon=50,
            max_state_dim=32,
            max_action_dim=32,
            dtype="float32",
        )

    @pytest.fixture
    def policy(self, config: Pi0Config) -> Pi0:
        """Create a Pi0 policy for testing."""
        return Pi0(
            variant=config.variant,
            paligemma_variant=config.paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            chunk_size=config.action_horizon,
            env_action_dim=config.action_dim,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
        )

    def test_initialization_and_components(self, policy: Pi0) -> None:
        """Test model initialization and key components."""
        assert isinstance(policy.model, Pi0Model)
        model = policy.model
        # Model no longer has config attribute (uses explicit args)
        assert all(
            hasattr(model, attr)
            for attr in ["state_proj", "action_in_proj", "action_out_proj", "paligemma_with_expert"]
        )

    def test_pi0_vs_pi05_differences(self) -> None:
        """Test that Pi0 and Pi0.5 have different components."""
        pi0_policy = Pi0(
            variant="pi0",
            paligemma_variant="gemma_300m",
            action_expert_variant="gemma_300m",
            chunk_size=50,
            env_action_dim=7,
        )
        pi05_policy = Pi0(
            variant="pi05",
            paligemma_variant="gemma_300m",
            action_expert_variant="gemma_300m",
            chunk_size=50,
            env_action_dim=7,
        )

        # Model no longer has config, check variant directly
        assert pi05_policy.model is not None
        assert pi0_policy.model is not None
        assert pi05_policy.model.is_pi05
        assert pi05_policy.model.use_adarms
        assert not pi0_policy.model.is_pi05
        assert not pi0_policy.model.use_adarms

    def test_configure_optimizers(self, policy: Pi0) -> None:
        """Test optimizer configuration."""
        # configure_optimizers needs trainer attached, so we test it can be called
        # In unit tests, trainer might not be attached, so we check the method exists
        assert hasattr(policy, "configure_optimizers")
        # Try to call it - it will fail if trainer not attached, but that's expected in unit tests
        try:
            result = policy.configure_optimizers()
            assert result is not None
        except RuntimeError as e:
            if "not attached to a `Trainer`" in str(e):
                # Expected in unit tests without trainer
                pass
            else:
                raise

    def test_model_from_config(self, config: Pi0Config) -> None:
        """Test model can be created from config (using explicit args)."""
        # Model no longer has from_config, create with explicit args
        model = Pi0Model(
            variant=config.variant,
            paligemma_variant=config.paligemma_variant,  # type: ignore[arg-type]
            action_expert_variant=config.action_expert_variant,  # type: ignore[arg-type]
            max_action_dim=config.max_action_dim,
            max_state_dim=config.max_state_dim,
            action_horizon=config.action_horizon,
            num_inference_steps=config.num_inference_steps,
            dtype=config.dtype,
        )
        assert isinstance(model, Pi0Model)
        # Verify model has correct variant
        assert model.variant == config.variant
