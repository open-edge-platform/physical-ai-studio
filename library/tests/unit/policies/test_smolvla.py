# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SmolVLA policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads).
"""

from __future__ import annotations

import unittest.mock

import pytest
import torch
from physicalai.config import Config
from physicalai.policies.smolvla import SmolVLA, SmolVLAConfig

# ============================================================================ #
# Configuration Tests                                                          #
# ============================================================================ #


class TestSmolVLAConfig:
    """Tests for SmolVLAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SmolVLAConfig()
        assert config.vlm_model_name == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SmolVLAConfig(
            chunk_size=100,
            n_action_steps=50,
            optimizer_lr=2e-4,
            freeze_vision_encoder=False,
            num_vlm_layers=8,
        )
        assert config.chunk_size == 100
        assert config.n_action_steps == 50
        assert config.optimizer_lr == 2e-4
        assert config.freeze_vision_encoder is False
        assert config.num_vlm_layers == 8

    def test_training_config_values(self) -> None:
        """Test training-related configuration values."""
        config = SmolVLAConfig()
        assert config.optimizer_betas == (0.9, 0.95)
        assert config.optimizer_eps == 1e-8
        assert config.optimizer_weight_decay == 1e-10
        assert config.optimizer_grad_clip_norm == 10
        assert config.scheduler_warmup_steps == 1_000
        assert config.scheduler_decay_steps == 30_000
        assert config.scheduler_decay_lr == 2.5e-6

    def test_expert_config_values(self) -> None:
        """Test action expert configuration values."""
        config = SmolVLAConfig()
        assert config.num_expert_layers == -1
        assert config.num_vlm_layers == 16
        assert config.self_attn_every_n_layers == 2
        assert config.expert_width_multiplier == 0.75
        assert config.min_period == 4e-3
        assert config.max_period == 4.0

    def test_n_action_steps_validation(self) -> None:
        """Test n_action_steps cannot exceed chunk_size."""
        with pytest.raises(ValueError, match="chunk size is the upper bound"):
            SmolVLAConfig(chunk_size=50, n_action_steps=100)

    def test_inheritance_and_serialization(self) -> None:
        """Test config inherits from base Config and supports serialization."""
        config = SmolVLAConfig(chunk_size=100, optimizer_lr=2e-4)
        assert isinstance(config, Config)

        # to_dict / from_dict round-trip
        config_dict = config.to_dict()
        assert config_dict["chunk_size"] == 100
        assert config_dict["optimizer_lr"] == 2e-4

        restored = SmolVLAConfig.from_dict(config_dict)
        assert restored.chunk_size == 100
        assert restored.optimizer_lr == 2e-4


# ============================================================================ #
# Policy Tests                                                                 #
# ============================================================================ #


class TestSmolVLAPolicy:
    """Tests for SmolVLA Lightning policy wrapper."""

    def test_lazy_initialization(self) -> None:
        """Test lazy initialization doesn't create model."""
        policy = SmolVLA()
        assert policy.model is None

    def test_hyperparameters_saved(self) -> None:
        """Test hyperparameters are saved for checkpoint."""
        policy = SmolVLA(
            chunk_size=100,
            optimizer_lr=2e-4,
            freeze_vision_encoder=False,
        )
        assert policy.hparams.chunk_size == 100
        assert policy.hparams.optimizer_lr == 2e-4
        assert policy.hparams.freeze_vision_encoder is False
        # Config dict stored in hparams
        assert "config" in policy.hparams
        assert policy.hparams["config"]["chunk_size"] == 100

    def test_config_attribute(self) -> None:
        """Test SmolVLA policy has config attribute."""
        policy = SmolVLA(chunk_size=100, optimizer_lr=2e-4)

        assert policy.config is not None
        assert policy.config.chunk_size == 100
        assert policy.config.optimizer_lr == 2e-4

    def test_n_action_steps(self) -> None:
        """Test n_action_steps is correctly set."""
        policy = SmolVLA(n_action_steps=25, chunk_size=50)
        assert policy._n_action_steps == 25
        assert policy.config.n_action_steps == 25

    @pytest.mark.parametrize("method", ["forward", "predict_action_chunk"])
    def test_methods_raise_without_model(self, method: str) -> None:
        """Test methods raise ValueError if model not initialized."""
        from physicalai.data import Observation

        policy = SmolVLA()
        dummy_obs = Observation(state=torch.randn(1, 10))
        with pytest.raises(ValueError, match="not initialized"):
            getattr(policy, method)(dummy_obs)


# ============================================================================ #
# Preprocessor Tests                                                           #
# ============================================================================ #


class TestSmolVLAPreprocessor:
    """Tests for SmolVLA preprocessor functions."""

    def test_make_smolvla_preprocessors(self) -> None:
        """Test make_smolvla_preprocessors returns callables."""
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        preprocessor, postprocessor = make_smolvla_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            stats=None,
            image_resolution=(512, 512),
            max_token_len=48,
            token_pad_type="longest",
        )
        assert callable(preprocessor)
        assert callable(postprocessor)

    def test_preprocessor_is_nn_module(self) -> None:
        """Test that preprocessors are nn.Module instances."""
        from physicalai.policies.smolvla.preprocessor import (
            SmolVLAPostprocessor,
            SmolVLAPreprocessor,
        )
        from torch import nn

        preprocessor = SmolVLAPreprocessor()
        postprocessor = SmolVLAPostprocessor()

        assert isinstance(preprocessor, nn.Module)
        assert isinstance(postprocessor, nn.Module)

    def test_preprocessor_default_values(self) -> None:
        """Test preprocessor default configuration values."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor()

        assert preprocessor.max_state_dim == 32
        assert preprocessor.max_action_dim == 32
        assert preprocessor.image_resolution == (512, 512)
        assert preprocessor.max_token_len == 48
        assert preprocessor.tokenizer_name == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        assert preprocessor.padding == "longest"

    def test_preprocessor_custom_values(self) -> None:
        """Test preprocessor with custom configuration values."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            max_state_dim=64,
            max_action_dim=16,
            image_resolution=(256, 256),
            max_token_len=64,
            padding="max_length",
        )

        assert preprocessor.max_state_dim == 64
        assert preprocessor.max_action_dim == 16
        assert preprocessor.image_resolution == (256, 256)
        assert preprocessor.max_token_len == 64
        assert preprocessor.padding == "max_length"

    def test_newline_processor_adds_newline(self) -> None:
        """Test newline processor adds newline to task strings."""
        from physicalai.data.observation import TASK
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        batch = {TASK: "Pick up the object"}
        result = SmolVLAPreprocessor._newline_processor(batch)
        assert result[TASK] == "Pick up the object\n"

    def test_newline_processor_preserves_newline(self) -> None:
        """Test newline processor preserves existing newline."""
        from physicalai.data.observation import TASK
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        batch = {TASK: "Pick up the object\n"}
        result = SmolVLAPreprocessor._newline_processor(batch)
        assert result[TASK] == "Pick up the object\n"

    def test_newline_processor_handles_list(self) -> None:
        """Test newline processor handles list of strings."""
        from physicalai.data.observation import TASK
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        batch = {TASK: ["Task 1", "Task 2\n", "Task 3"]}
        result = SmolVLAPreprocessor._newline_processor(batch)
        assert result[TASK] == ["Task 1\n", "Task 2\n", "Task 3\n"]

    def test_newline_processor_handles_none(self) -> None:
        """Test newline processor handles None task."""
        from physicalai.data.observation import TASK
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        batch = {TASK: None}
        result = SmolVLAPreprocessor._newline_processor(batch)
        assert result[TASK] == "\n"

    def test_newline_processor_missing_task(self) -> None:
        """Test newline processor handles missing task key."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        batch = {"other_key": "value"}
        result = SmolVLAPreprocessor._newline_processor(batch)
        assert result == {"other_key": "value"}

    def test_resize_with_pad_shape(self) -> None:
        """Test resize_with_pad produces correct output shape."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        # Input image: batch=2, channels=3, height=480, width=640
        img = torch.randn(2, 3, 480, 640)
        result = SmolVLAPreprocessor._resize_with_pad(img, width=512, height=512)

        assert result.shape == (2, 3, 512, 512)

    def test_resize_with_pad_invalid_dims(self) -> None:
        """Test resize_with_pad raises error for wrong dimensions."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        # 3D tensor instead of 4D
        img = torch.randn(3, 480, 640)
        with pytest.raises(ValueError, match="expected"):
            SmolVLAPreprocessor._resize_with_pad(img, width=512, height=512)

    def test_resize_with_pad_preserves_batch(self) -> None:
        """Test resize_with_pad preserves batch dimension."""
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        for batch_size in [1, 4, 8]:
            img = torch.randn(batch_size, 3, 480, 640)
            result = SmolVLAPreprocessor._resize_with_pad(img, width=256, height=256)
            assert result.shape[0] == batch_size

    def test_postprocessor_identity_without_features(self) -> None:
        """Test postprocessor acts as identity without features."""
        from physicalai.data.observation import ACTION
        from physicalai.policies.smolvla.preprocessor import SmolVLAPostprocessor

        postprocessor = SmolVLAPostprocessor(features=None)
        action = torch.randn(2, 10, 7)
        batch = {ACTION: action}

        result = postprocessor(batch)
        torch.testing.assert_close(result[ACTION], action)


# ============================================================================ #
# Feature Normalization Tests                                                  #
# ============================================================================ #


class TestFeatureNormalization:
    """Tests for feature normalization in SmolVLA preprocessor."""

    def test_preprocessor_with_features(self) -> None:
        """Test preprocessor with feature configuration."""
        from physicalai.data import Feature, FeatureType, NormalizationParameters
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        features = {
            "state": Feature(
                name="state",
                ftype=FeatureType.STATE,
                shape=(10,),
                normalization_data=NormalizationParameters(
                    mean=[0.0] * 10,
                    std=[1.0] * 10,
                ),
            ),
        }
        preprocessor = SmolVLAPreprocessor(features=features)

        # Should have normalizer set
        assert preprocessor._state_action_normalizer is not None

    def test_postprocessor_with_features(self) -> None:
        """Test postprocessor with feature configuration."""
        from physicalai.data import Feature, FeatureType, NormalizationParameters
        from physicalai.policies.smolvla.preprocessor import SmolVLAPostprocessor

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
        postprocessor = SmolVLAPostprocessor(features=features)

        # Should have denormalizer set
        assert postprocessor._action_denormalizer is not None

    def test_make_preprocessors_with_stats(self) -> None:
        """Test make_smolvla_preprocessors with dataset statistics."""
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        stats: dict[str, dict[str, list[float] | str | tuple]] = {
            "observation.state": {
                "name": "observation.state",
                "shape": (10,),
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
            },
        }

        preprocessor, postprocessor = make_smolvla_preprocessors(
            max_state_dim=32,
            max_action_dim=32,
            stats=stats,
        )

        assert preprocessor is not None
        assert postprocessor is not None


# ============================================================================ #
# Attention Mode Tests                                                         #
# ============================================================================ #


class TestAttentionModes:
    """Tests for attention mode configuration."""

    def test_cross_attention_mode(self) -> None:
        """Test cross attention mode configuration."""
        config = SmolVLAConfig(attention_mode="cross_attn")
        assert config.attention_mode == "cross_attn"

    def test_prefix_length_default(self) -> None:
        """Test prefix length default value."""
        config = SmolVLAConfig()
        assert config.prefix_length == -1

    def test_custom_prefix_length(self) -> None:
        """Test custom prefix length."""
        config = SmolVLAConfig(prefix_length=32)
        assert config.prefix_length == 32


# ============================================================================ #
# Camera Rename Tests                                                          #
# ============================================================================ #


class TestCameraRenameMap:
    """Tests for camera rename_map functionality."""

    def test_config_rename_map_default_none(self) -> None:
        config = SmolVLAConfig()
        assert config.rename_map is None

    def test_config_rename_map_tuple_format(self) -> None:
        config = SmolVLAConfig(rename_map=(("top", "camera1"), ("wrist", "camera2")))
        assert config.rename_map == (("top", "camera1"), ("wrist", "camera2"))

    def test_config_rename_map_serialization(self) -> None:
        config = SmolVLAConfig(rename_map=(("top", "camera1"),))
        config_dict = config.to_dict()
        assert config_dict["rename_map"] == [["top", "camera1"]]
        restored = SmolVLAConfig.from_dict(config_dict)
        assert restored.rename_map == (("top", "camera1"),)

    def test_policy_rename_map_saved_in_hparams(self) -> None:
        policy = SmolVLA(rename_map=(("top", "camera1"),))
        assert policy.hparams["rename_map"] == (("top", "camera1"),)
        assert policy.config.rename_map == (("top", "camera1"),)

    def test_preprocessor_rename_map_default_none(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor()
        assert preprocessor.rename_map is None

    def test_preprocessor_rename_map_stored(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        rename_map = {"top": "camera1", "wrist": "camera2"}
        preprocessor = SmolVLAPreprocessor(rename_map=rename_map)
        assert preprocessor.rename_map == rename_map

    def test_apply_rename_map_renames_image_keys(self) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        rename_map = {"top": "camera1", "wrist": "camera2"}
        preprocessor = SmolVLAPreprocessor(rename_map=rename_map)

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"{IMAGES}.wrist": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top", f"{IMAGES}.wrist"],
        }

        result = preprocessor._apply_rename_map(batch)

        assert f"{IMAGES}.camera1" in result
        assert f"{IMAGES}.camera2" in result
        assert f"{IMAGES}.top" not in result
        assert f"{IMAGES}.wrist" not in result
        assert result[f"_{IMAGES}_keys"] == [f"{IMAGES}.camera1", f"{IMAGES}.camera2"]

    def test_apply_rename_map_preserves_unmapped_keys(self) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        rename_map = {"top": "camera1"}
        preprocessor = SmolVLAPreprocessor(rename_map=rename_map)

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"{IMAGES}.side": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top", f"{IMAGES}.side"],
        }

        result = preprocessor._apply_rename_map(batch)

        assert f"{IMAGES}.camera1" in result
        assert f"{IMAGES}.side" in result
        assert result[f"_{IMAGES}_keys"] == [f"{IMAGES}.camera1", f"{IMAGES}.side"]

    def test_apply_rename_map_renames_padding_masks(self) -> None:
        from physicalai.data.observation import EXTRA, IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        rename_map = {"top": "camera1"}
        preprocessor = SmolVLAPreprocessor(rename_map=rename_map)

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"{EXTRA}.{IMAGES}.top_padding_mask": torch.ones(2, dtype=torch.bool),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }

        result = preprocessor._apply_rename_map(batch)

        assert f"{EXTRA}.{IMAGES}.camera1_padding_mask" in result
        assert f"{EXTRA}.{IMAGES}.top_padding_mask" not in result

    def test_apply_rename_map_noop_when_empty(self) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(rename_map=None)

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }
        original_keys = set(batch.keys())

        result = preprocessor._apply_rename_map(batch)

        assert set(result.keys()) == original_keys


class TestRenameStats:
    """Tests for rename_stats helper function."""

    def test_rename_stats_renames_image_keys(self) -> None:
        from physicalai.policies.smolvla.preprocessor import rename_stats

        stats = {
            "observation.top": {"name": "top", "mean": [0.5], "std": [0.1]},
            "observation.state": {"name": "observation.state", "mean": [0.0], "std": [1.0]},
            "action": {"name": "action", "mean": [0.0], "std": [1.0]},
        }
        rename_map = {"top": "camera1"}
        result = rename_stats(stats, rename_map)

        assert "observation.camera1" in result
        assert "observation.top" not in result
        assert result["observation.camera1"]["name"] == "observation.camera1"
        assert result["observation.state"]["name"] == "observation.state"
        assert result["action"]["name"] == "action"

    def test_rename_stats_preserves_unmapped_keys(self) -> None:
        from physicalai.policies.smolvla.preprocessor import rename_stats

        stats = {
            "observation.state": {"name": "observation.state", "mean": [0.0], "std": [1.0]},
            "action": {"name": "action", "mean": [0.0], "std": [1.0]},
        }
        rename_map = {"top": "camera1"}
        result = rename_stats(stats, rename_map)

        assert "observation.state" in result
        assert "action" in result

    def test_rename_stats_empty_stats(self) -> None:
        from physicalai.policies.smolvla.preprocessor import rename_stats

        assert rename_stats({}, {"top": "camera1"}) == {}

    def test_rename_stats_does_not_mutate_original(self) -> None:
        from physicalai.policies.smolvla.preprocessor import rename_stats

        stats = {
            "observation.top": {"name": "top", "mean": [0.5], "std": [0.1]},
        }
        original_keys = set(stats.keys())
        rename_stats(stats, {"top": "camera1"})

        assert set(stats.keys()) == original_keys
        assert stats["observation.top"]["name"] == "top"


class TestCameraNameValidation:
    """Tests for camera name mismatch warning."""

    def test_validate_warns_on_mismatch(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            expected_camera_names={"camera1", "camera2"},
        )

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)

        assert "Camera name mismatch detected" in caplog.text
        assert "top" in caplog.text
        assert "camera1" in caplog.text

    def test_validate_no_warning_when_matching(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            expected_camera_names={"top"},
        )

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)

        assert "Camera name mismatch" not in caplog.text

    def test_validate_no_warning_when_no_expected_names(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(expected_camera_names=None)

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)

        assert "Camera name mismatch" not in caplog.text

    def test_validate_runs_only_once(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            expected_camera_names={"camera1"},
        )

        batch = {
            f"{IMAGES}.top": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.top"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)
            caplog.clear()
            preprocessor._validate_camera_names(batch)

        assert "Camera name mismatch" not in caplog.text

    def test_validate_warns_on_camera_count_mismatch(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            expected_camera_names={"camera1", "camera2", "camera3"},
            empty_cameras=0,
        )

        batch = {
            f"{IMAGES}.camera1": torch.randn(2, 3, 64, 64),
            f"{IMAGES}.camera2": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.camera1", f"{IMAGES}.camera2"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)

        assert "Camera count mismatch" in caplog.text
        assert "empty_cameras=1" in caplog.text

    def test_validate_no_count_warning_when_padded(self, caplog: pytest.LogCaptureFixture) -> None:
        from physicalai.data.observation import IMAGES
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(
            expected_camera_names={"camera1", "camera2", "camera3"},
            empty_cameras=1,
        )

        batch = {
            f"{IMAGES}.camera1": torch.randn(2, 3, 64, 64),
            f"{IMAGES}.camera2": torch.randn(2, 3, 64, 64),
            f"_{IMAGES}_keys": [f"{IMAGES}.camera1", f"{IMAGES}.camera2"],
        }

        import logging

        with caplog.at_level(logging.WARNING):
            preprocessor._validate_camera_names(batch)

        assert "Camera count mismatch" not in caplog.text

    """Tests for make_smolvla_preprocessors with rename_map."""

    def test_factory_passes_rename_map(self) -> None:
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        rename_map = {"top": "camera1"}
        preprocessor, _ = make_smolvla_preprocessors(rename_map=rename_map)
        assert preprocessor.rename_map == rename_map

    def test_factory_renames_stats(self) -> None:
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        stats: dict[str, dict[str, list[float] | str | tuple]] = {
            "observation.state": {
                "name": "observation.state",
                "shape": (10,),
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
            },
        }

        preprocessor, postprocessor = make_smolvla_preprocessors(
            stats=stats,
            rename_map={"top": "camera1"},
        )
        assert preprocessor is not None
        assert postprocessor is not None

    def test_factory_without_rename_map(self) -> None:
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        preprocessor, _ = make_smolvla_preprocessors()
        assert preprocessor.rename_map is None


class TestEmptyCameras:
    """Tests for empty_cameras padding behavior."""

    def test_config_empty_cameras_default(self) -> None:
        config = SmolVLAConfig()
        assert config.empty_cameras == 0

    def test_config_empty_cameras_custom(self) -> None:
        config = SmolVLAConfig(empty_cameras=2)
        assert config.empty_cameras == 2

    def test_preprocessor_stores_empty_cameras(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(empty_cameras=1)
        assert preprocessor.empty_cameras == 1

    def test_preprocess_images_returns_lists(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(image_resolution=(64, 64))
        batch = {
            "state": torch.randn(2, 6),
            "images.cam1": torch.rand(2, 3, 64, 64),
        }
        images, masks = preprocessor._preprocess_images(batch)
        assert isinstance(images, list)
        assert isinstance(masks, list)
        assert len(images) == 1
        assert len(masks) == 1

    def test_empty_cameras_appends_placeholders(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(image_resolution=(64, 64), empty_cameras=2)
        batch = {
            "state": torch.randn(2, 6),
            "task": "pick up the cube\n",
            "images.cam1": torch.rand(2, 3, 64, 64),
            "_images_keys": ["images.cam1"],
        }

        with unittest.mock.patch.object(
            preprocessor,
            "_tokenize",
            return_value=(torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 10, dtype=torch.bool)),
        ):
            result = preprocessor(batch)

        assert result["images"].shape[0] == 3  # 1 real + 2 empty
        assert result["image_masks"].shape[0] == 3

    def test_empty_cameras_placeholder_values(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(image_resolution=(64, 64), empty_cameras=1)
        batch = {
            "state": torch.randn(2, 6),
            "task": "pick up the cube\n",
            "images.cam1": torch.rand(2, 3, 64, 64),
            "_images_keys": ["images.cam1"],
        }

        with unittest.mock.patch.object(
            preprocessor,
            "_tokenize",
            return_value=(torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 10, dtype=torch.bool)),
        ):
            result = preprocessor(batch)

        real_mask = result["image_masks"][0]
        empty_mask = result["image_masks"][1]
        assert real_mask.all()
        assert not empty_mask.any()

        empty_img = result["images"][1]
        assert torch.allclose(empty_img, torch.ones_like(empty_img) * -1)

    def test_empty_cameras_zero_noop(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(image_resolution=(64, 64), empty_cameras=0)
        batch = {
            "state": torch.randn(2, 6),
            "task": "pick up the cube\n",
            "images.cam1": torch.rand(2, 3, 64, 64),
            "_images_keys": ["images.cam1"],
        }

        with unittest.mock.patch.object(
            preprocessor,
            "_tokenize",
            return_value=(torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 10, dtype=torch.bool)),
        ):
            result = preprocessor(batch)

        assert result["images"].shape[0] == 1

    def test_empty_cameras_no_images_returns_empty(self) -> None:
        from physicalai.policies.smolvla.preprocessor import SmolVLAPreprocessor

        preprocessor = SmolVLAPreprocessor(image_resolution=(64, 64), empty_cameras=2)
        batch = {
            "state": torch.randn(2, 6),
            "task": "pick up the cube\n",
            "_images_keys": [],
        }

        with unittest.mock.patch.object(
            preprocessor,
            "_tokenize",
            return_value=(torch.zeros(2, 10, dtype=torch.long), torch.ones(2, 10, dtype=torch.bool)),
        ):
            result = preprocessor(batch)

        assert result["images"].shape[0] == 0

    def test_factory_passes_empty_cameras(self) -> None:
        from physicalai.policies.smolvla.preprocessor import make_smolvla_preprocessors

        preprocessor, _ = make_smolvla_preprocessors(empty_cameras=2)
        assert preprocessor.empty_cameras == 2

    def test_policy_empty_cameras_in_hparams(self) -> None:
        policy = SmolVLA(empty_cameras=1)
        assert policy.config.empty_cameras == 1
        assert policy.hparams["empty_cameras"] == 1
