# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Groot policy.

Fast, self-contained tests with no external dependencies (no LeRobot, no Isaac-GR00T).
"""

from __future__ import annotations

import pytest
import torch
from getiaction.config import Config
from getiaction.policies.groot import Groot, GrootConfig

# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestGrootConfig:
    """Tests for GrootConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GrootConfig()
        assert config.chunk_size == 50
        assert config.n_action_steps == 50
        assert config.max_state_dim == 64
        assert config.max_action_dim == 32
        assert config.attn_implementation == "sdpa"
        assert not config.tune_llm
        assert not config.tune_visual
        assert config.tune_projector
        assert config.tune_diffusion_model
        assert config.learning_rate == 1e-4

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = GrootConfig(
            chunk_size=100,
            learning_rate=2e-4,
            tune_llm=True,
            warmup_ratio=0.1,
        )
        assert config.chunk_size == 100
        assert config.learning_rate == 2e-4
        assert config.tune_llm
        assert config.warmup_ratio == 0.1

    def test_action_horizon_property(self) -> None:
        """Test action_horizon property is alias for chunk_size."""
        config = GrootConfig(chunk_size=75)
        assert config.action_horizon == 75

    def test_inheritance_and_serialization(self) -> None:
        """Test config inherits from base Config and supports serialization."""
        config = GrootConfig(chunk_size=100, learning_rate=2e-4)
        assert isinstance(config, Config)

        # to_dict / from_dict round-trip
        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 100
        assert config_dict["learning_rate"] == 2e-4

        restored = GrootConfig.from_dict(config_dict)
        assert restored.chunk_size == 100
        assert restored.learning_rate == 2e-4


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestGrootPolicy:
    """Tests for Groot Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test lazy initialization doesn't create model."""
        policy = Groot()
        assert policy.model is None
        assert not policy._is_setup_complete

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        policy = Groot(chunk_size=100, learning_rate=2e-4, tune_diffusion_model=False)
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.learning_rate == 2e-4
        assert not policy.hparams.tune_diffusion_model
        # Config dict stored in hparams
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 100

    def test_from_config(self) -> None:
        """Test Groot policy can be created from config."""
        config = GrootConfig(chunk_size=100, learning_rate=2e-4)
        policy = Groot.from_config(config)

        assert policy.model is None  # Lazy initialization
        assert policy.config.chunk_size == 100
        assert policy.config.learning_rate == 2e-4

    @pytest.mark.parametrize("method", ["forward", "select_action"])
    def test_methods_raise_without_model(self, method: str) -> None:
        """Test methods raise RuntimeError if model not initialized."""
        policy = Groot()
        with pytest.raises(RuntimeError, match="not initialized"):
            getattr(policy, method)({})

    def test_configure_optimizers_raises_without_model(self) -> None:
        """Test configure_optimizers raises RuntimeError if model not initialized."""
        policy = Groot()
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.configure_optimizers()


# ============================================================================ #
# NN Component Tests                                                           #
# ============================================================================ #


class TestNNPrimitives:
    """Tests for neural network primitives (nn.py)."""

    def test_swish_activation(self) -> None:
        """Test swish computes x * sigmoid(x) and preserves gradients."""
        from getiaction.policies.groot.components.nn import swish

        x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        out = swish(x)
        expected = x.detach() * torch.sigmoid(x.detach())
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        assert x.grad is not None

    def test_sinusoidal_encoding(self) -> None:
        """Test sinusoidal positional encoding produces unique, deterministic outputs."""
        from getiaction.policies.groot.components.nn import SinusoidalPositionalEncoding

        encoder = SinusoidalPositionalEncoding(embedding_dim=128)
        t = torch.tensor([[0.0, 100.0, 500.0]])

        out = encoder(t)
        assert out.shape == (1, 3, 128)

        # Deterministic
        torch.testing.assert_close(encoder(t), out)

        # Different timesteps → different encodings
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_category_specific_linear(self) -> None:
        """Test category-specific linear layer routes by category ID."""
        from getiaction.policies.groot.components.nn import CategorySpecificLinear

        layer = CategorySpecificLinear(num_categories=4, input_dim=16, hidden_dim=32)
        x = torch.randn(2, 5, 16, requires_grad=True)

        out1 = layer(x, torch.tensor([0, 0]))
        out2 = layer(x, torch.tensor([1, 1]))

        assert out1.shape == (2, 5, 32)
        assert not torch.allclose(out1, out2)  # Different categories
        out1.sum().backward()
        assert x.grad is not None

    def test_multi_embodiment_action_encoder(self) -> None:
        """Test multi-embodiment action encoder combines actions and timesteps."""
        from getiaction.policies.groot.components.nn import MultiEmbodimentActionEncoder

        encoder = MultiEmbodimentActionEncoder(action_dim=7, hidden_size=64, num_embodiments=4)
        actions = torch.randn(2, 10, 7)
        emb_ids = torch.tensor([0, 2])

        out1 = encoder(actions, torch.tensor([0, 0]), emb_ids)
        out2 = encoder(actions, torch.tensor([500, 500]), emb_ids)

        assert out1.shape == (2, 10, 64)
        assert not torch.allclose(out1, out2)  # Different timesteps


# ============================================================================ #
# Transformer Component Tests                                                  #
# ============================================================================ #


class TestTransformerComponents:
    """Tests for transformer components (transformer.py)."""

    def test_timestep_encoder(self) -> None:
        """Test timestep encoder produces unique embeddings per timestep."""
        from getiaction.policies.groot.components.transformer import TimestepEncoder

        encoder = TimestepEncoder(embedding_dim=256)

        out1 = encoder(torch.tensor([0]))
        out2 = encoder(torch.tensor([500]))

        assert out1.shape == (1, 256)
        assert not torch.allclose(out1, out2)

    def test_ada_layer_norm(self) -> None:
        """Test adaptive layer norm modulates based on timestep embedding."""
        from getiaction.policies.groot.components.transformer import AdaLayerNorm

        norm = AdaLayerNorm(embedding_dim=256)
        x = torch.randn(2, 16, 256)

        out1 = norm(x, torch.randn(2, 256))
        out2 = norm(x, torch.randn(2, 256))

        assert out1.shape == x.shape
        assert not torch.allclose(out1, out2)  # Different conditioning

    def test_basic_transformer_block(self) -> None:
        """Test basic transformer block with cross-attention."""
        from getiaction.policies.groot.components.transformer import BasicTransformerBlock

        block = BasicTransformerBlock(
            dim=256,
            num_attention_heads=4,
            attention_head_dim=64,
            cross_attention_dim=256,
        )
        block.eval()

        hidden = torch.randn(2, 16, 256, requires_grad=True)
        encoder_hidden = torch.randn(2, 50, 256)
        temb = torch.randn(2, 256)

        out = block(hidden, encoder_hidden_states=encoder_hidden, temb=temb)
        assert out.shape == hidden.shape
        out.sum().backward()
        assert hidden.grad is not None

    def test_dit(self) -> None:
        """Test Diffusion Transformer (DiT) forward pass."""
        from getiaction.policies.groot.components.transformer import get_dit_class

        DiT = get_dit_class()
        dit = DiT(num_attention_heads=4, attention_head_dim=64, num_layers=2, output_dim=256)
        dit.eval()

        hidden = torch.randn(2, 16, 256)
        encoder_hidden = torch.randn(2, 50, 256)
        timesteps = torch.tensor([100, 500])

        out = dit(hidden, encoder_hidden, timesteps)
        assert out.shape == (2, 16, 256)

    def test_self_attention_transformer(self) -> None:
        """Test self-attention transformer (VL encoder)."""
        from getiaction.policies.groot.components.transformer import (
            get_self_attention_transformer_class,
        )

        VL = get_self_attention_transformer_class()
        transformer = VL(num_attention_heads=4, attention_head_dim=64, num_layers=1, output_dim=256)
        transformer.eval()

        hidden = torch.randn(2, 50, 256)
        out = transformer(hidden)
        assert out.shape == hidden.shape


# ============================================================================ #
# Action Head Tests                                                            #
# ============================================================================ #


class TestFlowMatchingActionHead:
    """Tests for FlowMatchingActionHead."""

    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        from getiaction.policies.groot.components.action_head import FlowMatchingActionHeadConfig

        config = FlowMatchingActionHeadConfig(action_dim=32, action_horizon=16)
        assert config.action_dim == 32
        assert config.action_horizon == 16
        assert config.hidden_size == 1024
        assert config.max_num_embodiments == 32
        assert config.tune_projector
        assert config.tune_diffusion_model

    def test_from_config_and_from_dict(self) -> None:
        """Test FlowMatchingActionHead can be created from config or dict."""
        from getiaction.policies.groot.components.action_head import (
            FlowMatchingActionHead,
            FlowMatchingActionHeadConfig,
        )

        config = FlowMatchingActionHeadConfig(action_dim=8, action_horizon=16, hidden_size=128)
        head_from_config = FlowMatchingActionHead.from_config(config)
        head_from_dict = FlowMatchingActionHead.from_dict(
            {"action_dim": 8, "action_horizon": 16, "hidden_size": 128},
        )

        for head in [head_from_config, head_from_dict]:
            assert isinstance(head, FlowMatchingActionHead)
            assert head.action_dim == 8
            assert head.action_horizon == 16
            assert head.hidden_size == 128


# ============================================================================ #
# Preprocessor Tests                                                           #
# ============================================================================ #


class TestPreprocessor:
    """Tests for Groot preprocessor functions."""

    def test_make_groot_transforms(self) -> None:
        """Test make_groot_transforms returns callables."""
        from getiaction.policies.groot.transforms import make_groot_transforms

        preprocessor, postprocessor = make_groot_transforms(
            max_state_dim=64,
            max_action_dim=32,
            action_horizon=16,
            embodiment_tag="test",
            env_action_dim=7,
            stats=None,
            eagle_processor_repo="lerobot/eagle2hg-processor-groot-n1p5",
        )
        assert callable(preprocessor)
        assert callable(postprocessor)

    def test_preprocessor_is_nn_module(self) -> None:
        """Test that preprocessors are nn.Module instances."""
        from getiaction.policies.groot.transforms import GrootPostprocessor, GrootPreprocessor
        from torch import nn

        preprocessor = GrootPreprocessor()
        postprocessor = GrootPostprocessor()

        assert isinstance(preprocessor, nn.Module)
        assert isinstance(postprocessor, nn.Module)

    def test_preprocessor_buffers_registered(self) -> None:
        """Test that stats are registered as buffers."""
        from getiaction.policies.groot.transforms import GrootPreprocessor

        stats = {
            "observation.state": {"min": [0.0, 1.0], "max": [1.0, 2.0]},
            "action": {"min": [-1.0, -2.0], "max": [1.0, 2.0]},
        }
        preprocessor = GrootPreprocessor(stats=stats)

        # Check buffers are registered
        buffer_names = [name for name, _ in preprocessor.named_buffers()]
        assert "state_min" in buffer_names
        assert "state_max" in buffer_names
        assert "action_min" in buffer_names
        assert "action_max" in buffer_names

    def test_preprocessor_device_handling(self) -> None:
        """Test that preprocessor buffers move with .to(device)."""
        from getiaction.policies.groot.transforms import GrootPreprocessor

        stats = {
            "observation.state": {"min": [0.0], "max": [1.0]},
            "action": {"min": [-1.0], "max": [1.0]},
        }
        preprocessor = GrootPreprocessor(stats=stats)

        # Initially on CPU
        assert preprocessor.state_min.device.type == "cpu"

        # Move to CPU explicitly (always works)
        preprocessor = preprocessor.to("cpu")
        assert preprocessor.state_min.device.type == "cpu"
        assert preprocessor.action_min.device.type == "cpu"

    def test_postprocessor_buffers_registered(self) -> None:
        """Test that postprocessor stats are registered as buffers."""
        from getiaction.policies.groot.transforms import GrootPostprocessor

        stats = {"action": {"min": [-1.0, -2.0], "max": [1.0, 2.0]}}
        postprocessor = GrootPostprocessor(env_action_dim=2, stats=stats)

        buffer_names = [name for name, _ in postprocessor.named_buffers()]
        assert "action_min" in buffer_names
        assert "action_max" in buffer_names

    def test_preprocessor_state_normalization(self) -> None:
        """Test state normalization with registered buffers."""
        from getiaction.policies.groot.transforms import GrootPreprocessor

        stats = {
            "observation.state": {"min": [0.0, 0.0], "max": [10.0, 10.0]},
        }
        preprocessor = GrootPreprocessor(max_state_dim=4, stats=stats)

        # Input state in [0, 10] should normalize to [-1, 1]
        state = torch.tensor([[5.0, 5.0]])  # Middle value
        batch = {"observation.state": state}

        result = preprocessor(batch)

        # Middle of [0, 10] maps to 0.0 in [-1, 1]
        assert result["state"][0, 0, 0] == pytest.approx(0.0, abs=1e-5)
        assert result["state"][0, 0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_postprocessor_denormalization(self) -> None:
        """Test action denormalization with registered buffers."""
        from getiaction.policies.groot.transforms import GrootPostprocessor

        stats = {"action": {"min": [0.0, 0.0], "max": [10.0, 10.0]}}
        postprocessor = GrootPostprocessor(env_action_dim=2, stats=stats)

        # Normalized action 0.0 should denormalize to middle (5.0)
        normalized = torch.tensor([[0.0, 0.0]])
        result = postprocessor(normalized)

        assert result[0, 0] == pytest.approx(5.0, abs=1e-5)
        assert result[0, 1] == pytest.approx(5.0, abs=1e-5)
