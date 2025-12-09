# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Groot policy components.

Fast, self-contained tests with no external dependencies (no LeRobot, no Isaac-GR00T).
Tests cover:
- Neural network primitives (nn.py)
- Transformer components (transformer.py)
- Action head (action_head.py)
- Policy wrapper (policy.py)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

# =============================================================================
# NN PRIMITIVES (nn.py)
# =============================================================================


class TestSwish:
    """Test swish activation function."""

    def test_swish_output_shape(self) -> None:
        """Test swish preserves input shape."""
        from getiaction.policies.groot.components.nn import swish

        x = torch.randn(2, 16, 512)
        out = swish(x)
        assert out.shape == x.shape

    def test_swish_values(self) -> None:
        """Test swish computes x * sigmoid(x)."""
        from getiaction.policies.groot.components.nn import swish

        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        expected = x * torch.sigmoid(x)
        torch.testing.assert_close(swish(x), expected)

    def test_swish_gradient(self) -> None:
        """Test swish is differentiable."""
        from getiaction.policies.groot.components.nn import swish

        x = torch.randn(4, requires_grad=True)
        out = swish(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSinusoidalPositionalEncoding:
    """Test sinusoidal positional encoding."""

    def test_output_shape(self) -> None:
        """Test output shape is (B, T, D)."""
        from getiaction.policies.groot.components.nn import SinusoidalPositionalEncoding

        encoder = SinusoidalPositionalEncoding(embedding_dim=256)
        timesteps = torch.tensor([[0.0, 100.0, 500.0], [50.0, 250.0, 750.0]])
        out = encoder(timesteps)
        assert out.shape == (2, 3, 256)

    def test_different_timesteps_different_encodings(self) -> None:
        """Test different timesteps produce different encodings."""
        from getiaction.policies.groot.components.nn import SinusoidalPositionalEncoding

        encoder = SinusoidalPositionalEncoding(embedding_dim=128)
        t1 = torch.tensor([[0.0]])
        t2 = torch.tensor([[500.0]])
        enc1 = encoder(t1)
        enc2 = encoder(t2)
        assert not torch.allclose(enc1, enc2)

    def test_deterministic(self) -> None:
        """Test encoding is deterministic."""
        from getiaction.policies.groot.components.nn import SinusoidalPositionalEncoding

        encoder = SinusoidalPositionalEncoding(embedding_dim=128)
        t = torch.tensor([[100.0, 200.0]])
        enc1 = encoder(t)
        enc2 = encoder(t)
        torch.testing.assert_close(enc1, enc2)


class TestCategorySpecificLinear:
    """Test category-specific linear layer."""

    @pytest.fixture
    def layer(self) -> nn.Module:
        from getiaction.policies.groot.components.nn import CategorySpecificLinear

        return CategorySpecificLinear(num_categories=4, input_dim=16, hidden_dim=32)

    def test_output_shape(self, layer: nn.Module) -> None:
        """Test output shape is (B, T, hidden_dim)."""
        x = torch.randn(2, 5, 16)
        cat_ids = torch.tensor([0, 2])
        out = layer(x, cat_ids)
        assert out.shape == (2, 5, 32)

    def test_different_categories_different_outputs(self, layer: nn.Module) -> None:
        """Test different category IDs produce different outputs."""
        x = torch.randn(2, 5, 16)
        out1 = layer(x, torch.tensor([0, 0]))
        out2 = layer(x, torch.tensor([1, 1]))
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self, layer: nn.Module) -> None:
        """Test gradients flow through layer."""
        x = torch.randn(2, 5, 16, requires_grad=True)
        cat_ids = torch.tensor([0, 2])
        out = layer(x, cat_ids).sum()
        out.backward()
        assert x.grad is not None


class TestCategorySpecificMLP:
    """Test category-specific MLP."""

    @pytest.fixture
    def mlp(self) -> nn.Module:
        from getiaction.policies.groot.components.nn import CategorySpecificMLP

        return CategorySpecificMLP(
            num_categories=4,
            input_dim=16,
            hidden_dim=32,
            output_dim=24,
        )

    def test_output_shape(self, mlp: nn.Module) -> None:
        """Test output shape is (B, T, output_dim)."""
        x = torch.randn(2, 5, 16)
        cat_ids = torch.tensor([0, 2])
        out = mlp(x, cat_ids)
        assert out.shape == (2, 5, 24)

    def test_gradient_flow(self, mlp: nn.Module) -> None:
        """Test gradients flow through MLP."""
        x = torch.randn(2, 5, 16, requires_grad=True)
        cat_ids = torch.tensor([0, 2])
        out = mlp(x, cat_ids).sum()
        out.backward()
        assert x.grad is not None


class TestMultiEmbodimentActionEncoder:
    """Test multi-embodiment action encoder."""

    @pytest.fixture
    def encoder(self) -> nn.Module:
        from getiaction.policies.groot.components.nn import MultiEmbodimentActionEncoder

        return MultiEmbodimentActionEncoder(
            action_dim=7,
            hidden_size=64,
            num_embodiments=4,
        )

    def test_output_shape(self, encoder: nn.Module) -> None:
        """Test output shape is (B, T, hidden_size)."""
        actions = torch.randn(2, 10, 7)
        timesteps = torch.tensor([0, 500])
        emb_ids = torch.tensor([0, 2])
        out = encoder(actions, timesteps, emb_ids)
        assert out.shape == (2, 10, 64)

    def test_different_timesteps_different_outputs(self, encoder: nn.Module) -> None:
        """Test different timesteps produce different outputs."""
        actions = torch.randn(2, 10, 7)
        emb_ids = torch.tensor([0, 0])
        out1 = encoder(actions, torch.tensor([0, 0]), emb_ids)
        out2 = encoder(actions, torch.tensor([500, 500]), emb_ids)
        assert not torch.allclose(out1, out2)


# =============================================================================
# TRANSFORMER COMPONENTS (transformer.py)
# =============================================================================


class TestTimestepEncoder:
    """Test timestep encoder."""

    @pytest.fixture
    def encoder(self) -> nn.Module:
        from getiaction.policies.groot.components.transformer import TimestepEncoder

        return TimestepEncoder(embedding_dim=256)

    def test_output_shape(self, encoder: nn.Module) -> None:
        """Test output shape is (B, D)."""
        timesteps = torch.tensor([100, 500])
        out = encoder(timesteps)
        assert out.shape == (2, 256)

    def test_different_timesteps_different_outputs(self, encoder: nn.Module) -> None:
        """Test different timesteps produce different outputs."""
        out1 = encoder(torch.tensor([0]))
        out2 = encoder(torch.tensor([500]))
        assert not torch.allclose(out1, out2)


class TestAdaLayerNorm:
    """Test adaptive layer normalization."""

    @pytest.fixture
    def norm(self) -> nn.Module:
        from getiaction.policies.groot.components.transformer import AdaLayerNorm

        return AdaLayerNorm(embedding_dim=256)

    def test_output_shape(self, norm: nn.Module) -> None:
        """Test output shape matches input."""
        x = torch.randn(2, 16, 256)
        temb = torch.randn(2, 256)
        out = norm(x, temb)
        assert out.shape == x.shape

    def test_different_temb_different_outputs(self, norm: nn.Module) -> None:
        """Test different timestep embeddings produce different outputs."""
        x = torch.randn(2, 16, 256)
        out1 = norm(x, torch.randn(2, 256))
        out2 = norm(x, torch.randn(2, 256))
        assert not torch.allclose(out1, out2)


class TestBasicTransformerBlock:
    """Test basic transformer block with cross-attention."""

    @pytest.fixture
    def block(self) -> nn.Module:
        from getiaction.policies.groot.components.transformer import (
            BasicTransformerBlock,
        )

        return BasicTransformerBlock(
            dim=256,
            num_attention_heads=4,
            attention_head_dim=64,
            cross_attention_dim=256,
        )

    def test_output_shape(self, block: nn.Module) -> None:
        """Test output shape matches hidden states."""
        hidden = torch.randn(2, 16, 256)
        encoder_hidden = torch.randn(2, 50, 256)
        temb = torch.randn(2, 256)
        block.eval()
        out = block(hidden, encoder_hidden_states=encoder_hidden, temb=temb)
        assert out.shape == hidden.shape

    def test_gradient_flow(self, block: nn.Module) -> None:
        """Test gradients flow through block."""
        hidden = torch.randn(2, 16, 256, requires_grad=True)
        encoder_hidden = torch.randn(2, 50, 256)
        temb = torch.randn(2, 256)
        out = block(hidden, encoder_hidden_states=encoder_hidden, temb=temb).sum()
        out.backward()
        assert hidden.grad is not None


class TestDiT:
    """Test Diffusion Transformer."""

    @pytest.fixture
    def dit(self) -> nn.Module:
        from getiaction.policies.groot.components.transformer import get_dit_class

        DiT = get_dit_class()
        return DiT(
            num_attention_heads=4,
            attention_head_dim=64,
            num_layers=2,
            output_dim=256,
        )

    def test_output_shape(self, dit: nn.Module) -> None:
        """Test output shape is (B, T, output_dim)."""
        hidden = torch.randn(2, 16, 256)
        encoder_hidden = torch.randn(2, 50, 256)
        timesteps = torch.tensor([100, 500])
        dit.eval()
        out = dit(hidden, encoder_hidden, timesteps)
        assert out.shape == (2, 16, 256)


class TestSelfAttentionTransformer:
    """Test self-attention transformer (VL)."""

    @pytest.fixture
    def transformer(self) -> nn.Module:
        from getiaction.policies.groot.components.transformer import (
            get_self_attention_transformer_class,
        )

        VL = get_self_attention_transformer_class()
        return VL(
            num_attention_heads=4,
            attention_head_dim=64,
            num_layers=1,
            output_dim=256,
        )

    def test_output_shape(self, transformer: nn.Module) -> None:
        """Test output shape matches input."""
        hidden = torch.randn(2, 50, 256)
        transformer.eval()
        out = transformer(hidden)
        assert out.shape == hidden.shape


# =============================================================================
# ACTION HEAD (action_head.py)
# =============================================================================


class TestFlowMatchingActionHeadConfig:
    """Test FlowMatchingActionHeadConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values with required args."""
        from getiaction.policies.groot.components.action_head import (
            FlowMatchingActionHeadConfig,
        )

        config = FlowMatchingActionHeadConfig(action_dim=32, action_horizon=16)
        # Required values
        assert config.action_dim == 32
        assert config.action_horizon == 16
        # Default values
        assert config.hidden_size == 1024
        assert config.max_num_embodiments == 32
        assert config.max_state_dim == 64
        assert config.tune_projector is True
        assert config.tune_diffusion_model is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from getiaction.policies.groot.components.action_head import (
            FlowMatchingActionHeadConfig,
        )

        config = FlowMatchingActionHeadConfig(
            action_horizon=32,
            action_dim=64,
            num_inference_timesteps=20,
        )
        assert config.action_horizon == 32
        assert config.action_dim == 64
        assert config.num_inference_timesteps == 20

    def test_from_config_with_dataclass(self) -> None:
        """Test FromConfig mixin works with FlowMatchingActionHeadConfig dataclass."""
        from getiaction.policies.groot.components.action_head import (
            FlowMatchingActionHead,
            FlowMatchingActionHeadConfig,
        )

        config = FlowMatchingActionHeadConfig(
            action_dim=8,
            action_horizon=16,
            hidden_size=128,
        )
        # Use FromConfig.from_config to instantiate from dataclass
        head = FlowMatchingActionHead.from_config(config)
        assert isinstance(head, FlowMatchingActionHead)
        assert head.action_dim == 8
        assert head.action_horizon == 16
        assert head.hidden_size == 128

    def test_from_dict(self) -> None:
        """Test FromConfig mixin works with dict."""
        from getiaction.policies.groot.components.action_head import (
            FlowMatchingActionHead,
        )

        config_dict = {
            "action_dim": 8,
            "action_horizon": 16,
            "hidden_size": 128,
        }
        # Use FromConfig.from_dict to instantiate from dict
        head = FlowMatchingActionHead.from_dict(config_dict)
        assert isinstance(head, FlowMatchingActionHead)
        assert head.action_dim == 8
        assert head.action_horizon == 16
        assert head.hidden_size == 128


# =============================================================================
# POLICY (policy.py)
# =============================================================================


class TestGrootPolicy:
    """Test Groot Lightning policy wrapper."""

    def test_lazy_init_no_model(self) -> None:
        """Test lazy initialization doesn't create model."""
        from getiaction.policies.groot import Groot

        policy = Groot()
        assert policy.model is None
        assert policy._is_setup_complete is False

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        from getiaction.policies.groot import Groot

        policy = Groot(
            chunk_size=100,
            learning_rate=2e-4,
            tune_diffusion_model=False,
        )
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.learning_rate == 2e-4
        assert policy.hparams.tune_diffusion_model is False

    def test_default_hyperparameters(self) -> None:
        """Test default hyperparameter values."""
        from getiaction.policies.groot import Groot

        policy = Groot()
        assert policy.hparams.chunk_size == 50
        assert policy.hparams.n_action_steps == 50
        assert policy.hparams.max_state_dim == 64
        assert policy.hparams.max_action_dim == 32
        assert policy.hparams.attn_implementation == "sdpa"
        assert policy.hparams.tune_llm is False
        assert policy.hparams.tune_visual is False
        assert policy.hparams.tune_projector is True
        assert policy.hparams.tune_diffusion_model is True

    def test_configure_optimizers_raises_without_model(self) -> None:
        """Test configure_optimizers raises if model not initialized."""
        from getiaction.policies.groot import Groot

        policy = Groot()
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.configure_optimizers()

    def test_forward_raises_without_model(self) -> None:
        """Test forward raises if model not initialized."""
        from getiaction.policies.groot import Groot

        policy = Groot()
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.forward({})

    def test_select_action_raises_without_model(self) -> None:
        """Test select_action raises if model not initialized."""
        from getiaction.policies.groot import Groot

        policy = Groot()
        with pytest.raises(RuntimeError, match="not initialized"):
            policy.select_action({})


class TestGrootPolicySerializeStats:
    """Test stats serialization for checkpointing."""

    def test_serialize_stats_with_tensors(self) -> None:
        """Test _serialize_stats converts tensors to lists."""
        from getiaction.policies.groot import Groot

        stats = {
            "action": {
                "mean": torch.tensor([1.0, 2.0, 3.0]),
                "std": torch.tensor([0.1, 0.2, 0.3]),
            },
            "state": {
                "min": torch.tensor([-1.0]),
                "max": torch.tensor([1.0]),
            },
        }
        serialized = Groot._serialize_stats(stats)

        assert serialized is not None
        assert serialized["action"]["mean"] == pytest.approx([1.0, 2.0, 3.0])
        assert serialized["action"]["std"] == pytest.approx([0.1, 0.2, 0.3])
        assert serialized["state"]["min"] == pytest.approx([-1.0])
        assert serialized["state"]["max"] == pytest.approx([1.0])

    def test_serialize_stats_none(self) -> None:
        """Test _serialize_stats handles None."""
        from getiaction.policies.groot import Groot

        assert Groot._serialize_stats(None) is None


# =============================================================================
# PREPROCESSOR (preprocessor.py)
# =============================================================================


class TestPreprocessor:
    """Test Groot preprocessor functions."""

    def test_make_groot_preprocessors_returns_callables(self) -> None:
        """Test make_groot_preprocessors returns two callables."""
        from getiaction.policies.groot.preprocessor import make_groot_preprocessors

        preprocessor, postprocessor = make_groot_preprocessors(
            max_state_dim=64,
            max_action_dim=32,
            action_horizon=16,
            embodiment_tag="test_embodiment",
            env_action_dim=7,
            stats=None,
            eagle_processor_repo="lerobot/eagle2hg-processor-groot-n1p5",
        )
        assert callable(preprocessor)
        assert callable(postprocessor)
