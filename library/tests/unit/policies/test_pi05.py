# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pi05 policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads).
"""

from __future__ import annotations

import pytest
import torch
from physicalai.config import Config
from physicalai.policies.pi05 import Pi05, Pi05Config, Pi05Model
from physicalai.policies.pi05.pretrained_utils import (
    convert_normalization_stats,
    fix_state_dict_keys,
    parse_config_features,
    resolve_feature_shape,
)


# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestPi05Config:
    """Tests for Pi05Config dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Pi05Config()
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_300m"
        assert config.dtype == "float32"
        assert config.n_obs_steps == 1
        assert config.chunk_size == 50
        assert config.n_action_steps == 50

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = Pi05Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_2b",
            chunk_size=100,
            n_action_steps=50,
            optimizer_lr=1e-4,
            freeze_vision_encoder=True,
            train_expert_only=True,
        )
        assert config.paligemma_variant == "gemma_2b"
        assert config.action_expert_variant == "gemma_2b"
        assert config.chunk_size == 100
        assert config.n_action_steps == 50
        assert config.optimizer_lr == 1e-4
        assert config.freeze_vision_encoder is True
        assert config.train_expert_only is True

    def test_training_config_values(self) -> None:
        """Test training-related configuration values."""
        config = Pi05Config()
        assert config.optimizer_betas == (0.9, 0.95)
        assert config.optimizer_eps == 1e-8
        assert config.optimizer_weight_decay == 0.01
        assert config.optimizer_grad_clip_norm == 1.0
        assert config.scheduler_warmup_steps == 1_000
        assert config.scheduler_decay_steps == 30_000
        assert config.scheduler_decay_lr == 2.5e-6

    def test_flow_matching_config_values(self) -> None:
        """Test flow matching configuration values."""
        config = Pi05Config()
        assert config.num_inference_steps == 10
        assert config.time_sampling_beta_alpha == 1.5
        assert config.time_sampling_beta_beta == 1.0
        assert config.time_sampling_scale == 0.999
        assert config.time_sampling_offset == 0.001
        assert config.min_period == 4e-3
        assert config.max_period == 4.0

    def test_image_config_values(self) -> None:
        """Test image-related configuration values."""
        config = Pi05Config()
        assert config.image_resolution == (224, 224)
        assert config.empty_cameras == 0
        assert config.tokenizer_max_length == 200

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="n_action_steps"):
            Pi05Config(chunk_size=50, n_action_steps=100)

    def test_paligemma_variant_validation(self) -> None:
        """Test paligemma_variant must be valid."""
        with pytest.raises(ValueError, match="Invalid paligemma_variant"):
            Pi05Config(paligemma_variant="invalid")

    def test_action_expert_variant_validation(self) -> None:
        """Test action_expert_variant must be valid."""
        with pytest.raises(ValueError, match="Invalid action_expert_variant"):
            Pi05Config(action_expert_variant="invalid")

    def test_dtype_validation(self) -> None:
        """Test dtype must be valid."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            Pi05Config(dtype="float16")

    def test_valid_paligemma_variants(self) -> None:
        """Test valid paligemma variants are accepted."""
        for variant in ("gemma_300m", "gemma_2b"):
            config = Pi05Config(paligemma_variant=variant)
            assert config.paligemma_variant == variant

    def test_valid_action_expert_variants(self) -> None:
        """Test valid action expert variants are accepted."""
        for variant in ("gemma_300m", "gemma_2b"):
            config = Pi05Config(action_expert_variant=variant)
            assert config.action_expert_variant == variant

    def test_valid_dtypes(self) -> None:
        """Test valid dtypes are accepted."""
        for dtype in ("bfloat16", "float32"):
            config = Pi05Config(dtype=dtype)
            assert config.dtype == dtype

    def test_inheritance_and_serialization(self) -> None:
        """Test config inherits from base Config and supports serialization."""
        config = Pi05Config(chunk_size=100, optimizer_lr=1e-4)
        assert isinstance(config, Config)

        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 100
        assert config_dict["optimizer_lr"] == 1e-4

        restored = Pi05Config.from_dict(config_dict)
        assert restored.chunk_size == 100
        assert restored.optimizer_lr == 1e-4

    def test_frozen_dataclass(self) -> None:
        """Test Pi05Config is frozen (immutable)."""
        config = Pi05Config()
        with pytest.raises(AttributeError):
            config.chunk_size = 100  # type: ignore[misc]


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestPi05Policy:
    """Tests for Pi05 Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test lazy initialization doesn't create model."""
        policy = Pi05()
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        policy = Pi05(
            chunk_size=100,
            optimizer_lr=1e-4,
            freeze_vision_encoder=True,
        )
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.optimizer_lr == 1e-4
        assert policy.hparams.freeze_vision_encoder is True
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 100

    def test_config_attribute(self) -> None:
        """Test Pi05 policy has config attribute."""
        policy = Pi05(chunk_size=100, optimizer_lr=1e-4)

        assert policy.config is not None
        assert policy.config.chunk_size == 100
        assert policy.config.optimizer_lr == 1e-4

    def test_n_action_steps(self) -> None:
        """Test n_action_steps is correctly set."""
        policy = Pi05(n_action_steps=25, chunk_size=50)
        assert policy._n_action_steps == 25
        assert policy.config.n_action_steps == 25

    def test_dataset_stats_none_by_default(self) -> None:
        """Test dataset_stats is None when not provided."""
        policy = Pi05()
        assert policy._dataset_stats is None

    def test_preprocessor_none_by_default(self) -> None:
        """Test preprocessors are None before initialization."""
        policy = Pi05()
        assert policy._preprocessor is None
        assert policy._postprocessor is None

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        """Test methods raise ValueError if model not initialized."""
        from physicalai.data import Observation

        policy = Pi05()
        dummy_obs = Observation(state=torch.randn(1, 10))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(dummy_obs)

    def test_eager_initialization_with_stats(self) -> None:
        """Test eager initialization with dataset_stats creates model."""
        stats = {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
            },
        }
        policy = Pi05(dataset_stats=stats)
        assert policy.model is not None
        assert isinstance(policy.model, Pi05Model)
        assert policy._preprocessor is not None
        assert policy._postprocessor is not None

    def test_config_passed_through(self) -> None:
        """Test all config parameters are passed through to config object."""
        policy = Pi05(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="float32",
            num_inference_steps=20,
            image_resolution=(224, 224),
            gradient_checkpointing=True,
        )
        assert policy.config.paligemma_variant == "gemma_2b"
        assert policy.config.action_expert_variant == "gemma_300m"
        assert policy.config.dtype == "float32"
        assert policy.config.num_inference_steps == 20
        assert policy.config.image_resolution == (224, 224)
        assert policy.config.gradient_checkpointing is True


# ============================================================================ #
# Model Utility Tests                                                          #
# ============================================================================ #


class TestModelUtilities:
    """Tests for Pi05 model utility functions."""

    def test_pad_vector_shorter(self) -> None:
        """Test pad_vector pads shorter vectors."""
        from physicalai.policies.pi05.preprocessor import pad_vector

        v = torch.randn(2, 7)
        padded = pad_vector(v, 32)
        assert padded.shape == (2, 32)
        torch.testing.assert_close(padded[:, :7], v)
        assert (padded[:, 7:] == 0).all()

    def test_pad_vector_equal(self) -> None:
        """Test pad_vector with equal dimensions (no-op)."""
        from physicalai.policies.pi05.preprocessor import pad_vector

        v = torch.randn(2, 32)
        padded = pad_vector(v, 32)
        assert padded.shape == (2, 32)
        torch.testing.assert_close(padded, v)

    def test_pad_vector_longer(self) -> None:
        """Test pad_vector with longer vector (no-op)."""
        from physicalai.policies.pi05.preprocessor import pad_vector

        v = torch.randn(2, 64)
        padded = pad_vector(v, 32)
        assert padded.shape == (2, 64)

    def test_resize_with_pad_shape(self) -> None:
        """Test resize_with_pad produces correct output shape."""
        from physicalai.policies.pi05.preprocessor import resize_with_pad_torch

        img = torch.rand(2, 480, 640, 3)  # [B, H, W, C]
        result = resize_with_pad_torch(img, 224, 224)
        assert result.shape == (2, 224, 224, 3)

    def test_resize_with_pad_channels_first(self) -> None:
        """Test resize_with_pad with channels-first format."""
        from physicalai.policies.pi05.preprocessor import resize_with_pad_torch

        img = torch.rand(2, 3, 480, 640)  # [B, C, H, W]
        result = resize_with_pad_torch(img, 224, 224)
        assert result.shape == (2, 3, 224, 224)

    def test_resize_with_pad_3d(self) -> None:
        """Test resize_with_pad with 3D input (no batch dim)."""
        from physicalai.policies.pi05.preprocessor import resize_with_pad_torch

        img = torch.rand(480, 640, 3)  # [H, W, C]
        result = resize_with_pad_torch(img, 224, 224)
        assert result.shape == (1, 224, 224, 3)

    def test_resize_with_pad_uint8(self) -> None:
        """Test resize_with_pad with uint8 images."""
        from physicalai.policies.pi05.preprocessor import resize_with_pad_torch

        img = torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8)
        result = resize_with_pad_torch(img, 224, 224)
        assert result.dtype == torch.uint8
        assert result.shape == (2, 224, 224, 3)

    def test_resize_with_pad_unsupported_dtype(self) -> None:
        """Test resize_with_pad raises error for unsupported dtype."""
        from physicalai.policies.pi05.preprocessor import resize_with_pad_torch

        img = torch.rand(2, 480, 640, 3).to(torch.float16)
        with pytest.raises(ValueError, match="Unsupported image dtype"):
            resize_with_pad_torch(img, 224, 224)

    def test_create_sinusoidal_pos_embedding_shape(self) -> None:
        """Test sinusoidal positional embedding has correct shape."""
        from physicalai.policies.pi05.model import create_sinusoidal_pos_embedding

        time = torch.tensor([0.1, 0.5, 0.9])
        emb = create_sinusoidal_pos_embedding(time, 64, min_period=4e-3, max_period=4.0, device=time.device)
        assert emb.shape == (3, 64)

    def test_create_sinusoidal_pos_embedding_odd_dim(self) -> None:
        """Test sinusoidal embedding raises for odd dimension."""
        from physicalai.policies.pi05.model import create_sinusoidal_pos_embedding

        time = torch.tensor([0.5])
        with pytest.raises(ValueError, match="divisible by 2"):
            create_sinusoidal_pos_embedding(time, 65, min_period=4e-3, max_period=4.0, device=time.device)

    def test_create_sinusoidal_pos_embedding_wrong_ndim(self) -> None:
        """Test sinusoidal embedding raises for wrong ndim."""
        from physicalai.policies.pi05.model import create_sinusoidal_pos_embedding

        time = torch.tensor([[0.5]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="batch_size"):
            create_sinusoidal_pos_embedding(time, 64, min_period=4e-3, max_period=4.0, device=time.device)

    def test_sample_beta(self) -> None:
        """Test sample_beta returns correct shape."""
        from physicalai.policies.pi05.model import sample_beta

        result = sample_beta(1.5, 1.0, 4, torch.device("cpu"))
        assert result.shape == (4,)
        assert (result >= 0).all() and (result <= 1).all()

    def test_make_att_2d_masks_shape(self) -> None:
        """Test make_att_2d_masks returns correct shape."""
        from physicalai.policies.pi05.model import make_att_2d_masks

        pad_masks = torch.ones(2, 10, dtype=torch.bool)
        att_masks = torch.zeros(2, 10, dtype=torch.bool)
        result = make_att_2d_masks(pad_masks, att_masks)
        assert result.shape == (2, 10, 10)
        assert result.dtype == torch.bool

    def test_make_att_2d_masks_wrong_ndim(self) -> None:
        """Test make_att_2d_masks raises for wrong ndim."""
        from physicalai.policies.pi05.model import make_att_2d_masks

        pad_masks = torch.ones(2, 3, 10, dtype=torch.bool)  # 3D
        att_masks = torch.zeros(2, 10, dtype=torch.bool)
        with pytest.raises(ValueError, match="2D"):
            make_att_2d_masks(pad_masks, att_masks)

    def test_get_gemma_config_300m(self) -> None:
        """Test get_gemma_config for gemma_300m variant."""
        from physicalai.policies.pi05.model import get_gemma_config

        config = get_gemma_config("gemma_300m")
        assert config.width == 1024
        assert config.depth == 18
        assert config.mlp_dim == 4096
        assert config.num_heads == 8
        assert config.num_kv_heads == 1
        assert config.head_dim == 256

    def test_get_gemma_config_2b(self) -> None:
        """Test get_gemma_config for gemma_2b variant."""
        from physicalai.policies.pi05.model import get_gemma_config

        config = get_gemma_config("gemma_2b")
        assert config.width == 2048
        assert config.depth == 18
        assert config.mlp_dim == 16_384

    def test_get_gemma_config_unknown(self) -> None:
        """Test get_gemma_config raises for unknown variant."""
        from physicalai.policies.pi05.model import get_gemma_config

        with pytest.raises(ValueError, match="Unknown variant"):
            get_gemma_config("gemma_7b")

    def test_get_safe_dtype_cpu(self) -> None:
        """Test get_safe_dtype returns float32 for bfloat16 on CPU."""
        from physicalai.policies.pi05.model import _get_safe_dtype

        assert _get_safe_dtype(torch.bfloat16, "cpu") == torch.float32
        assert _get_safe_dtype(torch.float64, "cpu") == torch.float64
        assert _get_safe_dtype(torch.float32, "cpu") == torch.float32

    def test_get_safe_dtype_cuda(self) -> None:
        """Test get_safe_dtype returns target dtype for CUDA."""
        from physicalai.policies.pi05.model import _get_safe_dtype

        assert _get_safe_dtype(torch.bfloat16, "cuda") == torch.bfloat16
        assert _get_safe_dtype(torch.float32, "cuda") == torch.float32


# ============================================================================ #
# Pi Gemma Component Tests                                                     #
# ============================================================================ #


class TestPiGemmaComponents:
    """Tests for PiGemma model components."""

    def test_gated_residual_both_none(self) -> None:
        """Test gated_residual with both inputs None."""
        from physicalai.policies.pi05.pi_gemma import _gated_residual

        result = _gated_residual(None, None, None)
        assert result is None

    def test_gated_residual_x_none(self) -> None:
        """Test gated_residual with x None."""
        from physicalai.policies.pi05.pi_gemma import _gated_residual

        y = torch.randn(2, 3)
        result = _gated_residual(None, y, None)
        torch.testing.assert_close(result, y)

    def test_gated_residual_y_none(self) -> None:
        """Test gated_residual with y None."""
        from physicalai.policies.pi05.pi_gemma import _gated_residual

        x = torch.randn(2, 3)
        result = _gated_residual(x, None, None)
        torch.testing.assert_close(result, x)

    def test_gated_residual_no_gate(self) -> None:
        """Test gated_residual without gate (simple addition)."""
        from physicalai.policies.pi05.pi_gemma import _gated_residual

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        result = _gated_residual(x, y, None)
        torch.testing.assert_close(result, x + y)

    def test_gated_residual_with_gate(self) -> None:
        """Test gated_residual with gate modulation."""
        from physicalai.policies.pi05.pi_gemma import _gated_residual

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        gate = torch.randn(2, 3)
        result = _gated_residual(x, y, gate)
        torch.testing.assert_close(result, x + y * gate)

    def test_pi_gemma_rms_norm_standard(self) -> None:
        """Test PiGemmaRMSNorm without conditioning (standard mode)."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm

        norm = PiGemmaRMSNorm(dim=16)
        x = torch.randn(2, 10, 16)
        output, gate = norm(x)
        assert output.shape == x.shape
        assert gate is None

    def test_pi_gemma_rms_norm_adaptive(self) -> None:
        """Test PiGemmaRMSNorm with adaptive conditioning (AdaRMS)."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm

        norm = PiGemmaRMSNorm(dim=16, cond_dim=32)
        x = torch.randn(2, 10, 16)
        cond = torch.randn(2, 32)
        output, gate = norm(x, cond=cond)
        assert output.shape == x.shape
        assert gate is not None
        assert gate.shape == (2, 1, 16)  # Unsqueezed for 3D input

    def test_pi_gemma_rms_norm_cond_dim_mismatch(self) -> None:
        """Test PiGemmaRMSNorm raises for wrong cond dimension."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm

        norm = PiGemmaRMSNorm(dim=16, cond_dim=32)
        x = torch.randn(2, 10, 16)
        cond = torch.randn(2, 64)  # Wrong dim (expected 32)
        with pytest.raises(ValueError, match="Expected cond dim"):
            norm(x, cond=cond)

    def test_pi_gemma_rms_norm_extra_repr(self) -> None:
        """Test PiGemmaRMSNorm extra_repr for both modes."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm

        standard_norm = PiGemmaRMSNorm(dim=16)
        assert "adaptive" not in standard_norm.extra_repr()

        adaptive_norm = PiGemmaRMSNorm(dim=16, cond_dim=32)
        assert "adaptive=True" in adaptive_norm.extra_repr()
        assert "cond_dim=32" in adaptive_norm.extra_repr()

    def test_layernorm_forward_without_cond(self) -> None:
        """Test layernorm_forward without conditioning."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm, layernorm_forward

        norm = PiGemmaRMSNorm(dim=16)
        x = torch.randn(2, 10, 16)
        output, gate = layernorm_forward(norm, x, cond=None)
        assert output.shape == x.shape
        assert gate is None

    def test_layernorm_forward_with_cond(self) -> None:
        """Test layernorm_forward with conditioning."""
        from physicalai.policies.pi05.pi_gemma import PiGemmaRMSNorm, layernorm_forward

        norm = PiGemmaRMSNorm(dim=16, cond_dim=32)
        x = torch.randn(2, 10, 16)
        cond = torch.randn(2, 32)
        output, gate = layernorm_forward(norm, x, cond=cond)
        assert output.shape == x.shape
        assert gate is not None


# ============================================================================ #
# Preprocessor Tests                                                           #
# ============================================================================ #


class TestPi05Preprocessor:
    """Tests for Pi05 preprocessor functions."""

    def test_make_pi05_preprocessors(self) -> None:
        """Test make_pi05_preprocessors returns callables."""
        from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors

        preprocessor, postprocessor = make_pi05_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            stats=None,
            image_resolution=(224, 224),
            max_token_len=200,
        )
        assert callable(preprocessor)
        assert callable(postprocessor)

    def test_preprocessor_is_nn_module(self) -> None:
        """Test that preprocessors are nn.Module instances."""
        from physicalai.policies.pi05.preprocessor import Pi05Postprocessor, Pi05Preprocessor

        preprocessor = Pi05Preprocessor()
        postprocessor = Pi05Postprocessor()

        assert isinstance(preprocessor, torch.nn.Module)
        assert isinstance(postprocessor, torch.nn.Module)

    def test_preprocessor_default_values(self) -> None:
        """Test preprocessor default configuration values."""
        from physicalai.policies.pi05.preprocessor import Pi05Preprocessor

        preprocessor = Pi05Preprocessor()

        assert preprocessor.max_state_dim == 32
        assert preprocessor.max_action_dim == 32
        assert preprocessor.image_resolution == (224, 224)
        assert preprocessor.max_token_len == 200
        assert preprocessor.tokenizer_name == "google/paligemma-3b-pt-224"
        assert preprocessor.empty_cameras == 0

    def test_preprocessor_custom_values(self) -> None:
        """Test preprocessor with custom configuration values."""
        from physicalai.policies.pi05.preprocessor import Pi05Preprocessor

        preprocessor = Pi05Preprocessor(
            max_state_dim=64,
            max_action_dim=16,
            image_resolution=(512, 512),
            max_token_len=300,
            empty_cameras=2,
        )

        assert preprocessor.max_state_dim == 64
        assert preprocessor.max_action_dim == 16
        assert preprocessor.image_resolution == (512, 512)
        assert preprocessor.max_token_len == 300
        assert preprocessor.empty_cameras == 2

    def test_postprocessor_identity_without_features(self) -> None:
        """Test postprocessor acts as identity without features."""
        from physicalai.data.observation import ACTION
        from physicalai.policies.pi05.preprocessor import Pi05Postprocessor

        postprocessor = Pi05Postprocessor(features=None)
        action = torch.randn(2, 50, 7)
        batch = {ACTION: action}

        result = postprocessor(batch)
        torch.testing.assert_close(result[ACTION], action)

    def test_postprocessor_without_action_key(self) -> None:
        """Test postprocessor handles missing action key gracefully."""
        from physicalai.policies.pi05.preprocessor import Pi05Postprocessor

        postprocessor = Pi05Postprocessor(features=None)
        batch = {"other_key": torch.randn(2, 10)}
        result = postprocessor(batch)
        assert "other_key" in result

    def test_make_preprocessors_with_stats(self) -> None:
        """Test make_pi05_preprocessors with dataset statistics."""
        from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors

        stats: dict[str, dict] = {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
            },
        }

        preprocessor, postprocessor = make_pi05_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            stats=stats,
            image_resolution=(224, 224),
            max_token_len=200,
        )

        assert preprocessor is not None
        assert postprocessor is not None

    def test_make_preprocessors_maps_observation_prefix(self) -> None:
        """Test that make_pi05_preprocessors strips 'observation.' prefix from names."""
        from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors

        stats = {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
            },
        }

        preprocessor, _ = make_pi05_preprocessors(stats=stats)
        # The normalizer should have "state" mapped (not "observation.state")
        assert hasattr(preprocessor, "_state_action_normalizer")


# ============================================================================ #
# Feature Normalization Tests                                                  #
# ============================================================================ #


class TestFeatureNormalization:
    """Tests for feature normalization in Pi05 preprocessor."""

    def test_preprocessor_with_features(self) -> None:
        """Test preprocessor with feature configuration."""
        from physicalai.data import Feature, FeatureType, NormalizationParameters
        from physicalai.policies.pi05.preprocessor import Pi05Preprocessor

        features = {
            "state": Feature(
                name="state",
                ftype=FeatureType.STATE,
                shape=(8,),
                normalization_data=NormalizationParameters(
                    mean=[0.0] * 8,
                    std=[1.0] * 8,
                ),
            ),
        }
        preprocessor = Pi05Preprocessor(features=features)
        assert preprocessor._state_action_normalizer is not None

    def test_postprocessor_with_features(self) -> None:
        """Test postprocessor with feature configuration."""
        from physicalai.data import Feature, FeatureType, NormalizationParameters
        from physicalai.policies.pi05.preprocessor import Pi05Postprocessor

        features = {
            "action": Feature(
                name="action",
                ftype=FeatureType.ACTION,
                shape=(7,),
                normalization_data=NormalizationParameters(
                    mean=[0.0] * 7,
                    std=[1.0] * 7,
                ),
            ),
        }
        postprocessor = Pi05Postprocessor(features=features)
        assert postprocessor._action_denormalizer is not None


# ============================================================================ #
# Policy Static Method Tests                                                   #
# ============================================================================ #


class TestPretrainedUtils:
    """Tests for pretrained_utils helper functions."""

    def test_convert_normalization_stats_mean_std(self) -> None:
        """Test _convert_normalization_stats with mean/std."""
        mean, std = convert_normalization_stats({
            "mean": [0.0, 1.0, 2.0],
            "std": [0.5, 1.0, 1.5],
        })
        assert mean == [0.0, 1.0, 2.0]
        assert std == [0.5, 1.0, 1.5]

    def test_convert_normalization_stats_quantiles(self) -> None:
        """Test _convert_normalization_stats with quantiles (q01/q99)."""
        mean, std = convert_normalization_stats({
            "q01": [-1.0, -2.0],
            "q99": [1.0, 2.0],
        })
        assert mean == [0.0, 0.0]
        assert std == [1.0, 2.0]

    def test_convert_normalization_stats_min_max(self) -> None:
        """Test _convert_normalization_stats with min/max."""
        mean, std = convert_normalization_stats({
            "min": [-1.0, -2.0],
            "max": [1.0, 2.0],
        })
        assert mean == [0.0, 0.0]
        assert std == [1.0, 2.0]

    def test_convert_normalization_stats_empty(self) -> None:
        """Test _convert_normalization_stats with no recognized stats."""
        mean, std = convert_normalization_stats({})
        assert mean is None
        assert std is None

    def test_convert_normalization_stats_quantiles_zero_range(self) -> None:
        """Test _convert_normalization_stats clamps std to 1e-8 for zero range."""
        mean, std = convert_normalization_stats({
            "q01": [5.0],
            "q99": [5.0],
        })
        assert mean == [5.0]
        assert std == [1e-8]

    def test_fix_state_dict_keys_strips_model_prefix(self) -> None:
        """Test _fix_state_dict_keys strips 'model.' prefix."""
        original = {"model.some_layer.weight": torch.randn(3, 3)}
        fixed = fix_state_dict_keys(original)
        assert "some_layer.weight" in fixed
        assert "model.some_layer.weight" not in fixed

    def test_fix_state_dict_keys_renames_action_time_mlp(self) -> None:
        """Test _fix_state_dict_keys renames action_time_mlp to time_mlp."""
        original = {
            "action_time_mlp_in.weight": torch.randn(3, 3),
            "action_time_mlp_out.bias": torch.randn(3),
        }
        fixed = fix_state_dict_keys(original)
        assert "time_mlp_in.weight" in fixed
        assert "time_mlp_out.bias" in fixed

    def test_fix_state_dict_keys_skips_state_proj(self) -> None:
        """Test _fix_state_dict_keys skips state_proj keys."""
        original = {"state_proj.weight": torch.randn(3, 3)}
        fixed = fix_state_dict_keys(original)
        assert "state_proj.weight" not in fixed

    def test_fix_state_dict_keys_skips_expert_layernorm(self) -> None:
        """Test _fix_state_dict_keys skips expert layernorm weight keys."""
        original = {
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.weight": torch.randn(16),
            "paligemma_with_expert.gemma_expert.model.layers.0.post_attention_layernorm.weight": torch.randn(16),
            "paligemma_with_expert.gemma_expert.model.norm.weight": torch.randn(16),
        }
        fixed = fix_state_dict_keys(original)
        for key in original:
            assert key not in fixed

    def test_fix_state_dict_keys_copies_lm_head(self) -> None:
        """Test _fix_state_dict_keys copies lm_head to embed_tokens."""
        weight = torch.randn(100, 2048)
        original = {
            "model.paligemma_with_expert.paligemma.lm_head.weight": weight,
        }
        fixed = fix_state_dict_keys(original)
        tied_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        assert tied_key in fixed
        torch.testing.assert_close(fixed[tied_key], weight)

    def test_parse_config_features(self) -> None:
        """Test _parse_config_features extracts features from config."""
        hf_config = {
            "input_features": {
                "observation.state": {"shape": [8]},
            },
            "output_features": {
                "action": {"shape": [7]},
            },
        }
        stats = parse_config_features(hf_config)
        assert "observation.state" in stats
        assert "action" in stats
        assert stats["observation.state"]["mean"] == [0.0] * 8
        assert stats["action"]["std"] == [1.0] * 7

    def test_parse_config_features_empty(self) -> None:
        """Test _parse_config_features with empty config."""
        stats = parse_config_features({})
        assert stats == {}

    def test_resolve_feature_shape_from_config(self) -> None:
        """Test _resolve_feature_shape uses config features."""
        hf_config = {
            "input_features": {
                "observation.state": {"shape": [8]},
            },
        }
        shape = resolve_feature_shape("observation.state", hf_config, {})
        assert shape == (8,)

    def test_resolve_feature_shape_from_stats(self) -> None:
        """Test _resolve_feature_shape infers shape from stats."""
        shape = resolve_feature_shape(
            "observation.state",
            {},
            {"mean": [0.0, 0.0, 0.0]},
        )
        assert shape == (3,)

    def test_resolve_feature_shape_fallback(self) -> None:
        """Test _resolve_feature_shape falls back to (1,)."""
        shape = resolve_feature_shape("unknown", {}, {})
        assert shape == (1,)
