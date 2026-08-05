# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SmolVLA policy.

Fast, self-contained tests with no external dependencies (no HuggingFace model downloads).
"""

from __future__ import annotations

import pytest
import torch
from physicalai.training_config import Config
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

    def test_save_hyperparameters_ignores_compile_model(self) -> None:
        """Test compile_model is excluded from saved hyperparameters."""
        policy = SmolVLA(compile_model=True)
        assert "compile_model" not in policy.hparams

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
        assert preprocessor.padding == "max_length"

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
# Sample Input Tests                                                           #
# ============================================================================ #


class TestSampleInput:
    """Tests for SmolVLA.sample_input visual-feature detection.

    Uses a lightweight stub instead of constructing the full model to keep
    these tests fast and free of HuggingFace downloads.
    """

    @staticmethod
    def _call_sample_input(dataset_stats: dict) -> dict:
        """Invoke the SmolVLA.sample_input property on a minimal stub."""
        from physicalai.policies.smolvla import SmolVLA, SmolVLAConfig

        class _InnerStub:
            def __init__(self) -> None:
                # sample_input only reads device from this module's parameters.
                self._model = torch.nn.Linear(1, 1)

        class _ModelStub:
            def __init__(self) -> None:
                self._model = _InnerStub()._model

        class _Stub:
            def __init__(self, stats: dict) -> None:
                self._dataset_stats = stats
                self.model = _ModelStub()
                self.config = SmolVLAConfig()

        stub = _Stub(dataset_stats)
        # inputs_schema is consumed by the base sample_input property.
        stub.inputs_schema = SmolVLA.inputs_schema.fget(stub)  # type: ignore[attr-defined]
        return SmolVLA.sample_input.fget(stub)  # type: ignore[attr-defined]

    def test_sample_input_single_visual_feature_with_image_in_id(self) -> None:
        """Single visual feature whose id contains 'image' produces IMAGES key."""
        from physicalai.data.observation import IMAGES, STATE

        stats = {
            "observation.state": {"name": "state", "shape": (10,), "type": "STATE"},
            "observation.image": {"name": "image", "shape": (3, 512, 512), "type": "VISUAL"},
        }
        sample_input = self._call_sample_input(stats)
        assert STATE in sample_input
        assert IMAGES in sample_input
        assert sample_input[STATE].shape == (1, 10)
        assert sample_input[IMAGES].shape == (1, 3, 512, 512)

    def test_sample_input_single_visual_feature_without_image_in_id(self) -> None:
        """Visual feature without 'image' in id is still detected via the 'type' field."""
        from physicalai.data.observation import IMAGES, STATE

        stats = {
            "observation.state": {"name": "state", "shape": (10,), "type": "STATE"},
            "observation.front_cam": {
                "name": "front_cam",
                "shape": (3, 512, 512),
                "type": "VISUAL",
            },
        }
        sample_input = self._call_sample_input(stats)
        assert STATE in sample_input
        assert IMAGES in sample_input
        assert sample_input[IMAGES].shape == (1, 3, 512, 512)

    def test_sample_input_multiple_visual_features_without_image_in_id(self) -> None:
        """Multiple visual features without 'image' in id produce per-feature IMAGES.<name> keys."""
        from physicalai.data.observation import IMAGES, STATE

        stats = {
            "observation.state": {"name": "state", "shape": (10,), "type": "STATE"},
            "observation.front_cam": {
                "name": "front_cam",
                "shape": (3, 512, 512),
                "type": "VISUAL",
            },
            "observation.wrist_cam": {
                "name": "wrist_cam",
                "shape": (3, 512, 512),
                "type": "VISUAL",
            },
        }
        sample_input = self._call_sample_input(stats)
        assert STATE in sample_input
        assert f"{IMAGES}.front_cam" in sample_input
        assert f"{IMAGES}.wrist_cam" in sample_input
        assert IMAGES not in sample_input


# ============================================================================ #
# Action Padding Mask                                                          #
# ============================================================================ #


class TestActionPaddingMask:
    """Regression tests for end-of-episode action padding in the training loss.

    LeRobot clamps action-chunk queries at episode boundaries (repeating the
    final action) and flags the clamped steps as ``action_is_pad``. Those steps
    must not contribute to the flow-matching loss, and must not count towards
    its denominator either.
    """

    @staticmethod
    def _compute_loss(
        losses: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        key_suffix: str = ".action_is_pad",
    ) -> float:
        """Run ``SmolVLAModel.compute_loss`` against a stubbed model.

        Args:
            losses: Per-element losses the inner flow-matching model should return,
                shaped ``(batch, chunk, action_dim)``.
            action_is_pad: Optional ``(batch, chunk)`` bool padding mask.
            key_suffix: Batch key the mask is stored under, relative to ``EXTRA``.
                Overridable so a wrong key can be exercised.

        Returns:
            The scalar loss produced by ``compute_loss``.
        """
        from types import SimpleNamespace

        from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
        from physicalai.data.observation import ACTION, EXTRA, IMAGES
        from physicalai.policies.smolvla.model import SmolVLAModel

        action_dim = losses.shape[-1]
        batch: dict = {
            IMAGES: None,
            IMAGE_MASKS: None,
            TOKENIZED_PROMPT: None,
            TOKENIZED_PROMPT_MASK: None,
        }
        if action_is_pad is not None:
            batch[EXTRA + key_suffix] = action_is_pad

        stub = SimpleNamespace(
            _preprocess_batch=lambda b: b,
            _prepare_state=lambda b: None,
            _prepare_action=lambda b: None,
            _model=SimpleNamespace(forward=lambda *_a, **_kw: losses.clone()),
            _dataset_stats={ACTION: {"shape": (action_dim,)}},
        )
        loss, _ = SmolVLAModel.compute_loss(stub, batch)
        return float(loss)

    def test_mask_is_read_from_lerobot_key_only(self) -> None:
        """The mask must be read as ``action_is_pad``, LeRobot's actual key.

        The lookup uses ``.get()``, so a typo'd key silently disables masking with
        no error. This pins the exact spelling that
        ``lerobot/datasets/dataset_reader.py`` emits.
        """
        losses = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [99.0, 99.0], [99.0, 99.0]]])
        action_is_pad = torch.tensor([[False, False, True, True]])

        correct_key = self._compute_loss(losses, action_is_pad)
        typo_key = self._compute_loss(losses, action_is_pad, key_suffix=".actions_id_pad")

        assert correct_key == pytest.approx(1.0), "mask under the LeRobot key must apply"
        assert typo_key == pytest.approx(50.0), "a wrong key must not silently half-apply"

    def test_padded_steps_are_excluded_from_the_loss(self) -> None:
        """Padded steps contribute nothing, regardless of their magnitude."""
        losses = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [99.0, 99.0], [99.0, 99.0]]])
        action_is_pad = torch.tensor([[False, False, True, True]])

        masked = self._compute_loss(losses, action_is_pad)
        unmasked = self._compute_loss(losses, None)

        assert masked == pytest.approx(1.0), "padded steps must not affect the loss"
        assert unmasked == pytest.approx(50.0), "without a mask the padding dominates"

    def test_denominator_counts_only_valid_steps(self) -> None:
        """The loss divides by valid elements, not by the full tensor.

        Chosen so all three behaviours are distinguishable:
        correct = 2.0, mask-with-plain-mean = 1.0, no-mask = 50.5.
        A plain ``.mean()`` over the zeroed tensor would scale the loss - and
        therefore the gradient - down by the padding fraction.
        """
        losses = torch.tensor([[[2.0, 2.0], [2.0, 2.0], [99.0, 99.0], [99.0, 99.0]]])
        action_is_pad = torch.tensor([[False, False, True, True]])

        masked = self._compute_loss(losses, action_is_pad)

        assert masked == pytest.approx(2.0)
        assert masked != pytest.approx(1.0), "denominator must exclude padded elements"
        assert masked != pytest.approx(50.5), "mask must be applied at all"

    def test_fully_padded_chunk_does_not_divide_by_zero(self) -> None:
        """An all-padded chunk clamps the denominator instead of producing NaN."""
        losses = torch.ones(1, 4, 2)
        action_is_pad = torch.ones(1, 4, dtype=torch.bool)

        masked = self._compute_loss(losses, action_is_pad)

        assert masked == pytest.approx(0.0)

    def test_no_mask_falls_back_to_plain_mean(self) -> None:
        """Batches without the key (e.g. non-chunked datasets) keep the old path."""
        losses = torch.full((2, 3, 2), 4.0)
        assert self._compute_loss(losses, None) == pytest.approx(4.0)

    def test_masking_is_autograd_safe(self) -> None:
        """Gradients flow, and padded steps receive exactly zero gradient.

        This masking branch never executed before the key fix, so the autograd
        behaviour was previously unverified.
        """
        from types import SimpleNamespace

        from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
        from physicalai.data.observation import ACTION, EXTRA, IMAGES
        from physicalai.policies.smolvla.model import SmolVLAModel

        source = torch.ones(1, 4, 2, requires_grad=True)
        action_is_pad = torch.tensor([[False, False, True, True]])

        stub = SimpleNamespace(
            _preprocess_batch=lambda b: b,
            _prepare_state=lambda b: None,
            _prepare_action=lambda b: None,
            _model=SimpleNamespace(forward=lambda *_a, **_kw: source * 2.0),
            _dataset_stats={ACTION: {"shape": (2,)}},
        )
        batch: dict = {
            IMAGES: None,
            IMAGE_MASKS: None,
            TOKENIZED_PROMPT: None,
            TOKENIZED_PROMPT_MASK: None,
            EXTRA + ".action_is_pad": action_is_pad,
        }

        loss, _ = SmolVLAModel.compute_loss(stub, batch)
        loss.backward()

        assert source.grad is not None
        assert torch.all(source.grad[0, :2] != 0), "valid steps must receive gradient"
        assert torch.all(source.grad[0, 2:] == 0), "padded steps must receive zero gradient"
