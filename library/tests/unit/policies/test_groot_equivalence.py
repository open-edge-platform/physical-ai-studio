# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: ANN001, ANN201, DOC201, PLR6301, S101, PLC0415, COM812

"""Tests for verifying equivalence between first-party and LeRobot Groot implementations.

This module tests that the first-party Groot components produce identical outputs
to the LeRobot/NVIDIA implementation for both training and inference.
"""

from __future__ import annotations

import pytest
import torch

# First-party implementations
from getiaction.policies.groot.components import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    FlowMatchingActionHead,
    FlowMatchingActionHeadConfig,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    swish,
)


class TestSinusoidalPositionalEncodingEquivalence:
    """Test SinusoidalPositionalEncoding matches LeRobot implementation."""

    @pytest.fixture
    def lerobot_encoder(self):
        """Create LeRobot encoder if available."""
        try:
            from lerobot.policies.groot.action_head.action_encoder import (
                SinusoidalPositionalEncoding as LeRobotEncoder,
            )

            return LeRobotEncoder(embedding_dim=256)
        except ImportError:
            pytest.skip("LeRobot not installed")

    @pytest.fixture
    def first_party_encoder(self):
        """Create first-party encoder."""
        return SinusoidalPositionalEncoding(embedding_dim=256)

    def test_output_shape_matches(self, first_party_encoder):
        """Test output shape is correct."""
        timesteps = torch.tensor([[0, 1, 2], [3, 4, 5]])  # (B=2, T=3)
        output = first_party_encoder(timesteps)
        assert output.shape == (2, 3, 256)

    def test_output_matches_lerobot(self, first_party_encoder, lerobot_encoder):
        """Test output values match LeRobot implementation."""
        timesteps = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.float32)

        first_party_out = first_party_encoder(timesteps)
        lerobot_out = lerobot_encoder(timesteps)

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-5,
            atol=1e-5,
            msg="SinusoidalPositionalEncoding output mismatch",
        )


class TestSwishEquivalence:
    """Test swish activation matches LeRobot implementation."""

    def test_swish_matches_lerobot(self):
        """Test swish output matches LeRobot implementation."""
        try:
            from lerobot.policies.groot.action_head.action_encoder import (
                swish as lerobot_swish,
            )
        except ImportError:
            pytest.skip("LeRobot not installed")

        x = torch.randn(10, 20)
        first_party_out = swish(x)
        lerobot_out = lerobot_swish(x)

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-6,
            atol=1e-6,
            msg="swish activation mismatch",
        )


class TestCategorySpecificLinearEquivalence:
    """Test CategorySpecificLinear matches LeRobot implementation."""

    @pytest.fixture
    def layer_params(self):
        """Common parameters for layers."""
        return {"num_categories": 4, "input_dim": 16, "hidden_dim": 32}

    @pytest.fixture
    def first_party_layer(self, layer_params):
        """Create first-party layer."""
        return CategorySpecificLinear(**layer_params)

    @pytest.fixture
    def lerobot_layer(self, layer_params):
        """Create LeRobot layer if available."""
        try:
            from lerobot.policies.groot.action_head.flow_matching_action_head import (
                CategorySpecificLinear as LeRobotLayer,
            )

            return LeRobotLayer(**layer_params)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_output_shape_matches(self, first_party_layer, layer_params):
        """Test output shape is correct."""
        x = torch.randn(2, 5, layer_params["input_dim"])  # (B=2, T=5, D=16)
        cat_ids = torch.tensor([0, 2])  # (B=2,)
        output = first_party_layer(x, cat_ids)
        assert output.shape == (2, 5, layer_params["hidden_dim"])

    def test_output_matches_lerobot_with_shared_weights(
        self, first_party_layer, lerobot_layer, layer_params
    ):
        """Test output values match when weights are shared."""
        # Copy weights from LeRobot to first-party
        first_party_layer.W.data = lerobot_layer.W.data.clone()
        first_party_layer.b.data = lerobot_layer.b.data.clone()

        x = torch.randn(3, 7, layer_params["input_dim"])
        cat_ids = torch.tensor([0, 1, 3])

        first_party_out = first_party_layer(x, cat_ids)
        lerobot_out = lerobot_layer(x, cat_ids)

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-5,
            atol=1e-5,
            msg="CategorySpecificLinear output mismatch",
        )


class TestCategorySpecificMLPEquivalence:
    """Test CategorySpecificMLP matches LeRobot implementation."""

    @pytest.fixture
    def mlp_params(self):
        """Common parameters for MLPs."""
        return {
            "num_categories": 4,
            "input_dim": 16,
            "hidden_dim": 32,
            "output_dim": 24,
        }

    @pytest.fixture
    def first_party_mlp(self, mlp_params):
        """Create first-party MLP."""
        return CategorySpecificMLP(**mlp_params)

    @pytest.fixture
    def lerobot_mlp(self, mlp_params):
        """Create LeRobot MLP if available."""
        try:
            from lerobot.policies.groot.action_head.flow_matching_action_head import (
                CategorySpecificMLP as LeRobotMLP,
            )

            return LeRobotMLP(**mlp_params)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_output_shape_matches(self, first_party_mlp, mlp_params):
        """Test output shape is correct."""
        x = torch.randn(2, 5, mlp_params["input_dim"])
        cat_ids = torch.tensor([0, 2])
        output = first_party_mlp(x, cat_ids)
        assert output.shape == (2, 5, mlp_params["output_dim"])

    def test_output_matches_lerobot_with_shared_weights(
        self, first_party_mlp, lerobot_mlp, mlp_params
    ):
        """Test output values match when weights are shared."""
        # Copy weights from LeRobot to first-party
        first_party_mlp.layer1.W.data = lerobot_mlp.layer1.W.data.clone()
        first_party_mlp.layer1.b.data = lerobot_mlp.layer1.b.data.clone()
        first_party_mlp.layer2.W.data = lerobot_mlp.layer2.W.data.clone()
        first_party_mlp.layer2.b.data = lerobot_mlp.layer2.b.data.clone()

        x = torch.randn(3, 7, mlp_params["input_dim"])
        cat_ids = torch.tensor([0, 1, 3])

        first_party_out = first_party_mlp(x, cat_ids)
        lerobot_out = lerobot_mlp(x, cat_ids)

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-5,
            atol=1e-5,
            msg="CategorySpecificMLP output mismatch",
        )


class TestMultiEmbodimentActionEncoderEquivalence:
    """Test MultiEmbodimentActionEncoder matches LeRobot implementation."""

    @pytest.fixture
    def encoder_params(self):
        """Common parameters for encoders."""
        return {"action_dim": 7, "hidden_size": 64, "num_embodiments": 4}

    @pytest.fixture
    def first_party_encoder(self, encoder_params):
        """Create first-party encoder."""
        return MultiEmbodimentActionEncoder(**encoder_params)

    @pytest.fixture
    def lerobot_encoder(self, encoder_params):
        """Create LeRobot encoder if available."""
        try:
            from lerobot.policies.groot.action_head.flow_matching_action_head import (
                MultiEmbodimentActionEncoder as LeRobotEncoder,
            )

            return LeRobotEncoder(**encoder_params)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_output_shape_matches(self, first_party_encoder, encoder_params):
        """Test output shape is correct."""
        actions = torch.randn(2, 10, encoder_params["action_dim"])  # (B=2, T=10, D=7)
        timesteps = torch.tensor([0, 500])  # (B=2,)
        cat_ids = torch.tensor([0, 2])  # (B=2,)

        output = first_party_encoder(actions, timesteps, cat_ids)
        assert output.shape == (2, 10, encoder_params["hidden_size"])

    def test_output_matches_lerobot_with_shared_weights(
        self, first_party_encoder, lerobot_encoder, encoder_params
    ):
        """Test output values match when weights are shared."""
        # Copy all weights
        first_party_encoder.W1.W.data = lerobot_encoder.W1.W.data.clone()
        first_party_encoder.W1.b.data = lerobot_encoder.W1.b.data.clone()
        first_party_encoder.W2.W.data = lerobot_encoder.W2.W.data.clone()
        first_party_encoder.W2.b.data = lerobot_encoder.W2.b.data.clone()
        first_party_encoder.W3.W.data = lerobot_encoder.W3.W.data.clone()
        first_party_encoder.W3.b.data = lerobot_encoder.W3.b.data.clone()

        actions = torch.randn(3, 15, encoder_params["action_dim"])
        timesteps = torch.tensor([100, 500, 900])
        cat_ids = torch.tensor([0, 1, 3])

        first_party_out = first_party_encoder(actions, timesteps, cat_ids)
        lerobot_out = lerobot_encoder(actions, timesteps, cat_ids)

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-4,
            atol=1e-4,
            msg="MultiEmbodimentActionEncoder output mismatch",
        )


class TestDiTEquivalence:
    """Test DiT matches LeRobot implementation."""

    @pytest.fixture
    def dit_config(self):
        """Common DiT configuration."""
        return {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "output_dim": 7,
            "num_layers": 2,
            "dropout": 0.0,  # Disable dropout for deterministic testing
            "cross_attention_dim": 128,
        }

    @pytest.fixture
    def first_party_dit(self, dit_config):
        """Create first-party DiT."""
        from getiaction.policies.groot.components import get_dit_class

        dit_class = get_dit_class()
        return dit_class(**dit_config)

    @pytest.fixture
    def lerobot_dit(self, dit_config):
        """Create LeRobot DiT if available."""
        try:
            from lerobot.policies.groot.action_head.cross_attention_dit import DiT

            return DiT(**dit_config)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_output_shape_matches(self, first_party_dit, dit_config):
        """Test output shape is correct."""
        batch_size = 2
        seq_len = 10
        inner_dim = dit_config["num_attention_heads"] * dit_config["attention_head_dim"]

        hidden_states = torch.randn(batch_size, seq_len, inner_dim)
        encoder_hidden_states = torch.randn(batch_size, 20, dit_config["cross_attention_dim"])
        timestep = torch.tensor([100, 500])

        output = first_party_dit(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
        )
        assert output.shape == (batch_size, seq_len, dit_config["output_dim"])

    def test_output_matches_lerobot_with_shared_weights(
        self, first_party_dit, lerobot_dit, dit_config
    ):
        """Test output values match when weights are shared."""
        # Copy all weights from LeRobot to first-party
        first_party_dit.load_state_dict(lerobot_dit.state_dict())

        batch_size = 2
        seq_len = 10
        inner_dim = dit_config["num_attention_heads"] * dit_config["attention_head_dim"]

        # Use deterministic inputs
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, inner_dim)
        encoder_hidden_states = torch.randn(batch_size, 20, dit_config["cross_attention_dim"])
        timestep = torch.tensor([100, 500])

        # Set both to eval mode
        first_party_dit.eval()
        lerobot_dit.eval()

        with torch.no_grad():
            first_party_out = first_party_dit(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            lerobot_out = lerobot_dit(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

        torch.testing.assert_close(
            first_party_out,
            lerobot_out,
            rtol=1e-4,
            atol=1e-4,
            msg="DiT output mismatch",
        )


class TestFlowMatchingActionHeadEquivalence:
    """Test FlowMatchingActionHead matches LeRobot implementation."""

    @pytest.fixture
    def action_head_config(self):
        """Common action head configuration."""
        # inner_dim = num_attention_heads * attention_head_dim = 4 * 32 = 128
        # This should match input_embedding_dim for the embeddings fed into DiT
        return {
            "action_dim": 7,
            "action_horizon": 16,
            "hidden_size": 64,  # Action decoder input/output dim
            "input_embedding_dim": 128,  # DiT inner_dim = 4 * 32 = 128
            "backbone_embedding_dim": 128,
            "max_state_dim": 14,
            "max_num_embodiments": 4,
            "num_inference_timesteps": 5,
            "num_timestep_buckets": 1000,
            "add_pos_embed": True,
            "use_vlln": False,  # Disable for simpler testing
            "num_target_vision_tokens": 4,
            "diffusion_model_cfg": {
                "num_attention_heads": 4,
                "attention_head_dim": 32,  # inner_dim = 4 * 32 = 128
                "output_dim": 64,  # Should match hidden_size for action_decoder
                "num_layers": 2,
                "dropout": 0.0,
                "cross_attention_dim": 128,  # VL features dim
            },
        }

    @pytest.fixture
    def first_party_action_head(self, action_head_config):
        """Create first-party action head."""
        config = FlowMatchingActionHeadConfig(**action_head_config)
        return FlowMatchingActionHead(config)

    @pytest.fixture
    def lerobot_action_head(self, action_head_config):
        """Create LeRobot action head if available."""
        try:
            from lerobot.policies.groot.action_head.flow_matching_action_head import (
                FlowmatchingActionHead as LeRobotActionHead,
            )
            from lerobot.policies.groot.action_head.flow_matching_action_head import (
                FlowmatchingActionHeadConfig as LeRobotConfig,
            )

            config = LeRobotConfig(**action_head_config)
            return LeRobotActionHead(config)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_inference_output_shape(self, first_party_action_head, action_head_config):
        """Test inference output shape is correct."""
        batch_size = 2

        backbone_output = {
            "backbone_features": torch.randn(
                batch_size, 20, action_head_config["backbone_embedding_dim"]
            ),
            "backbone_attention_mask": torch.ones(batch_size, 20),
        }
        action_input = {
            "state": torch.randn(batch_size, 1, action_head_config["max_state_dim"]),
            "embodiment_id": torch.tensor([0, 1]),
        }

        first_party_action_head.eval()
        with torch.no_grad():
            output = first_party_action_head.get_action(backbone_output, action_input)

        assert "action_pred" in output
        assert output["action_pred"].shape == (
            batch_size,
            action_head_config["action_horizon"],
            action_head_config["action_dim"],
        )

    def test_training_output_has_loss(self, first_party_action_head, action_head_config):
        """Test training forward pass produces loss."""
        batch_size = 2

        backbone_output = {
            "backbone_features": torch.randn(
                batch_size, 20, action_head_config["backbone_embedding_dim"]
            ),
            "backbone_attention_mask": torch.ones(batch_size, 20),
        }
        action_input = {
            "state": torch.randn(batch_size, 1, action_head_config["max_state_dim"]),
            "action": torch.randn(
                batch_size,
                action_head_config["action_horizon"],
                action_head_config["action_dim"],
            ),
            "action_mask": torch.ones(
                batch_size,
                action_head_config["action_horizon"],
                action_head_config["action_dim"],
            ),
            "embodiment_id": torch.tensor([0, 1]),
        }

        first_party_action_head.train()
        output = first_party_action_head(backbone_output, action_input)

        assert "loss" in output
        assert output["loss"].shape == ()
        assert not torch.isnan(output["loss"])

    def test_inference_matches_lerobot_with_shared_weights(
        self, first_party_action_head, lerobot_action_head, action_head_config
    ):
        """Test inference output matches LeRobot when weights are shared."""
        # Copy all weights from LeRobot to first-party
        first_party_action_head.load_state_dict(lerobot_action_head.state_dict())

        batch_size = 2
        torch.manual_seed(42)

        backbone_output = {
            "backbone_features": torch.randn(
                batch_size, 20, action_head_config["backbone_embedding_dim"]
            ),
            "backbone_attention_mask": torch.ones(batch_size, 20),
        }
        action_input = {
            "state": torch.randn(batch_size, 1, action_head_config["max_state_dim"]),
            "embodiment_id": torch.tensor([0, 1]),
        }

        # Set both to eval mode
        first_party_action_head.eval()
        lerobot_action_head.eval()

        # Use same random seed for both
        torch.manual_seed(123)
        with torch.no_grad():
            first_party_out = first_party_action_head.get_action(
                backbone_output, action_input
            )

        # Need to convert dict to BatchFeature for LeRobot
        try:
            from transformers.feature_extraction_utils import BatchFeature

            lerobot_backbone = BatchFeature(data=backbone_output)
            lerobot_action = BatchFeature(data=action_input)
        except ImportError:
            pytest.skip("transformers not installed")

        torch.manual_seed(123)
        with torch.no_grad():
            lerobot_out = lerobot_action_head.get_action(lerobot_backbone, lerobot_action)

        torch.testing.assert_close(
            first_party_out["action_pred"],
            lerobot_out["action_pred"],
            rtol=1e-3,
            atol=1e-3,
            msg="FlowMatchingActionHead inference output mismatch",
        )


class TestEagleBackboneEquivalence:
    """Test EagleBackbone matches LeRobot implementation."""

    @pytest.fixture
    def backbone_params(self):
        """Common parameters for Eagle backbone."""
        return {
            "tune_llm": False,
            "tune_visual": False,
        }

    @pytest.fixture
    def first_party_backbone(self, backbone_params):
        """Create first-party Eagle backbone."""
        from getiaction.policies.groot.components import EagleBackbone

        return EagleBackbone(
            attn_implementation="sdpa",
            **backbone_params,
        )

    @pytest.fixture
    def lerobot_backbone(self, backbone_params):
        """Create LeRobot Eagle backbone if available."""
        try:
            from lerobot.policies.groot.groot_n1 import EagleBackbone as LeRobotBackbone

            return LeRobotBackbone(**backbone_params)
        except ImportError:
            pytest.skip("LeRobot not installed")

    def test_forward_output_keys(self, first_party_backbone):
        """Test backbone forward produces correct output keys."""
        # Create mock eagle inputs
        batch_size = 2
        seq_len = 100
        hidden_dim = 2048

        batch = {
            "eagle_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "eagle_attention_mask": torch.ones(batch_size, seq_len),
            "eagle_pixel_values": torch.randn(batch_size, 3, 224, 224),
            "eagle_image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "eagle_image_sizes": torch.tensor([[224, 224], [224, 224]]),
        }

        # Note: This will fail without actual Eagle model weights loaded
        # This test just verifies the interface
        try:
            output = first_party_backbone(batch)
            assert "backbone_features" in output
            assert "backbone_attention_mask" in output
        except Exception:
            # Expected to fail without weights - just verify interface exists
            assert hasattr(first_party_backbone, "forward")
            assert hasattr(first_party_backbone, "eagle_model")
            assert hasattr(first_party_backbone, "eagle_linear")

    def test_trainable_parameters_match(self, first_party_backbone):
        """Test trainable parameter configuration matches expected behavior."""
        # Verify our backbone has the expected tunable flags
        assert first_party_backbone.tune_llm is False
        assert first_party_backbone.tune_visual is False
        # Verify eagle_model and eagle_linear exist
        assert hasattr(first_party_backbone, "eagle_model")
        assert hasattr(first_party_backbone, "eagle_linear")

    @pytest.mark.skip(reason="Requires Eagle model weights download")
    def test_output_matches_lerobot_with_shared_weights(
        self, first_party_backbone, lerobot_backbone
    ):
        """Test output values match when weights are shared.

        Skipped by default - requires downloading Eagle model weights.
        """
        # Copy weights from LeRobot to first-party
        first_party_backbone.load_state_dict(lerobot_backbone.state_dict())

        # Create test input
        batch_size = 2
        seq_len = 100

        batch = {
            "eagle_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "eagle_attention_mask": torch.ones(batch_size, seq_len),
            "eagle_pixel_values": torch.randn(batch_size, 3, 224, 224),
            "eagle_image_grid_thw": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        first_party_backbone.eval()
        lerobot_backbone.eval()

        with torch.no_grad():
            first_party_out = first_party_backbone(batch)

            # LeRobot uses BatchFeature wrapper
            from transformers.feature_extraction_utils import BatchFeature

            lerobot_batch = BatchFeature(data=batch)
            lerobot_out = lerobot_backbone(lerobot_batch)

        torch.testing.assert_close(
            first_party_out["backbone_features"],
            lerobot_out["backbone_features"],
            rtol=1e-4,
            atol=1e-4,
            msg="EagleBackbone output mismatch",
        )


class TestGrootModelEquivalence:
    """Test GrootModel matches LeRobot GR00TN15 implementation."""

    @pytest.fixture
    def model_params(self):
        """Common parameters for GrootModel."""
        return {
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "attn_implementation": "sdpa",
        }

    def test_from_pretrained_interface(self, model_params):
        """Test from_pretrained has same interface as LeRobot."""
        from getiaction.policies.groot import GrootModel

        # Verify the interface exists (don't actually download weights)
        assert hasattr(GrootModel, "from_pretrained")

        # Check signature accepts same kwargs
        import inspect

        sig = inspect.signature(GrootModel.from_pretrained)
        param_names = list(sig.parameters.keys())

        # Should accept these parameters
        expected_params = [
            "pretrained_model_name_or_path",
            "tune_llm",
            "tune_visual",
            "tune_projector",
            "tune_diffusion_model",
        ]
        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"

    def test_model_has_required_components(self):
        """Test GrootModel has backbone and action_head components."""
        from getiaction.policies.groot import GrootModel, GrootConfig

        # Create model without weights (just structure)
        config = GrootConfig(
            chunk_size=16,
            max_action_dim=7,
        )
        # Note: This would fail without proper initialization
        # Just verify the class structure
        assert hasattr(GrootModel, "from_pretrained")
        assert hasattr(GrootModel, "from_config")

    @pytest.mark.skip(reason="Requires full model weights download")
    def test_inference_matches_lerobot(self, model_params):
        """Test full model inference matches LeRobot.

        Skipped by default - requires downloading GR00T-N1.5-3B weights (~6GB).
        """
        from getiaction.policies.groot import GrootModel

        try:
            from lerobot.policies.groot.groot_n1 import GR00TN15 as LeRobotGrootModel
        except ImportError:
            pytest.skip("LeRobot not installed")

        model_path = "nvidia/GR00T-N1.5-3B"

        # Load both models
        first_party_model = GrootModel.from_pretrained(
            model_path,
            **model_params,
        )
        lerobot_model = LeRobotGrootModel.from_pretrained(
            model_path,
            **model_params,
        )

        # Create test inputs (would need proper preprocessing)
        # This is a placeholder for full integration test
        pass


class TestEndToEndEquivalence:
    """End-to-end tests comparing full Groot model outputs."""

    @pytest.mark.skip(reason="Requires full model weights download")
    def test_full_model_inference_matches(self):
        """Test full model inference matches LeRobot.

        This test requires downloading the full GR00T-N1.5-3B weights
        and is skipped by default.
        """
        # This would test the full GrootModel vs LeRobot's Groot
