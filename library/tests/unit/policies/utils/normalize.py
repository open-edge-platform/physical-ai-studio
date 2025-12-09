# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Policy Data Normalization"""

import numpy as np
import pytest
import torch
from torch import nn

from getiaction.data import Feature, FeatureType, NormalizationParameters
from getiaction.policies.utils.normalization import FeatureNormalizeTransform, NormalizationType


class TestFeatureNormalizeTransform:
    """Tests for FeatureNormalizeTransform class."""

    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return {
            "state": Feature(
                normalization_data=NormalizationParameters(mean=[1.0, 2.0], std=[0.5, 1.0]),
                shape=(2,),
                ftype=FeatureType.STATE
            ),
            "action": Feature(
                normalization_data=NormalizationParameters(min=[-1.0], max=[1.0]),
                shape=(1,),
                ftype=FeatureType.ACTION
            ),
            "image": Feature(
                normalization_data=NormalizationParameters(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
                shape=(3, 64, 64),
                ftype=FeatureType.VISUAL
            )
        }

    @pytest.fixture
    def norm_map(self):
        """Sample normalization mapping."""
        return {
            FeatureType.STATE: NormalizationType.MEAN_STD,
            FeatureType.ACTION: NormalizationType.MIN_MAX,
            FeatureType.VISUAL: NormalizationType.MEAN_STD
        }

    @pytest.fixture
    def sample_batch(self):
        """Sample batch for testing."""
        return {
            "state": torch.tensor([[1.5, 3.0], [2.0, 4.0]], dtype=torch.float32),
            "action": torch.tensor([[0.5], [-0.5]], dtype=torch.float32),
            "image": torch.randn(2, 3, 64, 64, dtype=torch.float32)
        }

    def test_initialization(self, sample_features, norm_map):
        """Test proper initialization of FeatureNormalizeTransform."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)

        assert transform._features == sample_features
        assert transform._norm_map == norm_map
        assert transform._inverse is False

        # Check that buffers are created
        assert hasattr(transform, "buffer_state")
        assert hasattr(transform, "buffer_action")
        assert hasattr(transform, "buffer_image")

    def test_initialization_inverse(self, sample_features, norm_map):
        """Test initialization with inverse=True."""
        transform = FeatureNormalizeTransform(sample_features, norm_map, inverse=True)
        assert transform._inverse is True

    def test_mean_std_normalization_forward(self, sample_features, norm_map, sample_batch):
        """Test MEAN_STD normalization in forward mode."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)
        batch = sample_batch.copy()

        original_state = batch["state"].clone()
        result = transform(batch)

        # Check that normalization was applied
        expected_mean = torch.tensor([1.0, 2.0], dtype=torch.float32)
        expected_std = torch.tensor([0.5, 1.0], dtype=torch.float32)
        expected_normalized = (original_state - expected_mean) / (expected_std + 1e-8)

        torch.testing.assert_close(result["state"], expected_normalized, rtol=1e-5, atol=1e-8)

    def test_mean_std_normalization_inverse(self, sample_features, norm_map):
        """Test MEAN_STD normalization in inverse mode."""
        transform = FeatureNormalizeTransform(sample_features, norm_map, inverse=True)

        # Use normalized data
        normalized_batch = {
            "state": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        }

        original_state = normalized_batch["state"].clone()
        result = transform(normalized_batch)

        # Check denormalization
        expected_mean = torch.tensor([1.0, 2.0], dtype=torch.float32)
        expected_std = torch.tensor([0.5, 1.0], dtype=torch.float32)
        expected_denormalized = original_state * expected_std + expected_mean

        torch.testing.assert_close(result["state"], expected_denormalized)

    def test_min_max_normalization_forward(self, sample_features, norm_map, sample_batch):
        """Test MIN_MAX normalization in forward mode."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)
        batch = sample_batch.copy()

        original_action = batch["action"].clone()
        result = transform(batch)

        # MIN_MAX: normalize to [0,1] then to [-1,1]
        min_val = torch.tensor([-1.0], dtype=torch.float32)
        max_val = torch.tensor([1.0], dtype=torch.float32)
        normalized_01 = (original_action - min_val) / (max_val - min_val + 1e-8)
        expected_normalized = normalized_01 * 2 - 1

        torch.testing.assert_close(result["action"], expected_normalized, rtol=1e-5, atol=1e-8)

    def test_min_max_normalization_inverse(self, sample_features, norm_map):
        """Test MIN_MAX normalization in inverse mode."""
        transform = FeatureNormalizeTransform(sample_features, norm_map, inverse=True)

        normalized_batch = {
            "action": torch.tensor([[0.0], [-1.0]], dtype=torch.float32)
        }

        original_state = normalized_batch["action"].clone()
        result = transform(normalized_batch)

        # Inverse MIN_MAX: from [-1,1] to [0,1] then to original range
        min_val = torch.tensor([-1.0], dtype=torch.float32)
        max_val = torch.tensor([1.0], dtype=torch.float32)
        from_neg1_1 = (original_state + 1) / 2
        expected_denormalized = from_neg1_1 * (max_val - min_val) + min_val

        torch.testing.assert_close(result["action"], expected_denormalized)

    def test_identity_normalization(self):
        """Test IDENTITY normalization does nothing."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(mean=0, std=1),
                shape=(2,),
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: NormalizationType.IDENTITY}

        transform = FeatureNormalizeTransform(features, norm_map)
        batch = {"state": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
        original = batch["state"].clone()

        result = transform(batch)
        torch.testing.assert_close(result["state"], original)

    def test_nested_batch_structure(self, sample_features, norm_map):
        """Test handling of nested batch dictionaries."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)

        nested_batch = {
            "observations": {
                "state": torch.tensor([[1.5, 3.0]], dtype=torch.float32)
            },
            "action": torch.tensor([[0.5]], dtype=torch.float32)
        }

        result = transform(nested_batch)

        # Should find and normalize nested state
        assert "observations" in result
        assert "state" in result["observations"]
        # Action should also be normalized
        assert "action" in result

    def test_visual_feature_shape_handling(self, sample_features, norm_map):
        """Test that visual features get proper shape handling."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)

        # Check that visual feature buffer has shape (3, 1, 1)
        buffer = transform.buffer_image
        assert buffer["mean"].shape == (3, 1, 1)
        assert buffer["std"].shape == (3, 1, 1)

    def test_create_stats_buffers_visual_validation(self):
        """Test validation of visual features in buffer creation."""
        # Invalid dimensions
        features = {
            "image": Feature(
                normalization_data=NormalizationParameters(mean=0, std=1),
                shape=(64, 64),  # Missing channel dimension
                ftype=FeatureType.VISUAL
            )
        }
        norm_map = {FeatureType.VISUAL: NormalizationType.MEAN_STD}

        with pytest.raises(ValueError, match="number of dimensions"):
            FeatureNormalizeTransform(features, norm_map)

        # Invalid channel-first format
        features = {
            "image": Feature(
                normalization_data=NormalizationParameters(mean=0, std=1),
                shape=(64, 64, 3),  # Channel-last format
                ftype=FeatureType.VISUAL
            )
        }

        with pytest.raises(ValueError, match="not channel first"):
            FeatureNormalizeTransform(features, norm_map)

    def test_create_stats_buffers_invalid_norm_mode(self):
        """Test error handling for invalid normalization mode."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(mean=0, std=1),
                shape=(2,),
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: "INVALID_MODE"}  # Invalid type

        with pytest.raises(TypeError, match="Invalid type of normalization mode"):
            FeatureNormalizeTransform(features, norm_map)

    def test_infinity_buffer_check(self, sample_features, norm_map):
        """Test infinity check in normalization buffers."""
        transform = FeatureNormalizeTransform(sample_features, norm_map)

        # Manually set buffer to infinity to trigger error
        transform.buffer_state["mean"].data.fill_(torch.inf)

        batch = {"state": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}

        with pytest.raises(ValueError, match="Normalization buffer 'mean' is infinity"):
            transform(batch)

    def test_list_conversion(self):
        """Test conversion of lists to torch tensors."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(
                    mean=[1.0, 2.0],
                    std=[0.5, 1.0]
                ),
                shape=(2,),
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: NormalizationType.MEAN_STD}

        transform = FeatureNormalizeTransform(features, norm_map)

        # Check that lists were converted to torch tensors
        assert isinstance(transform.buffer_state["mean"], nn.Parameter)
        assert isinstance(transform.buffer_state["std"], nn.Parameter)
        assert transform.buffer_state["mean"].dtype == torch.float32

    def test_integer_normalization_data(self):
        """Test handling of integer normalization data."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(mean=1, std=2),
                shape=(1,),
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: NormalizationType.MEAN_STD}

        transform = FeatureNormalizeTransform(features, norm_map)

        # Check that integers were converted to torch tensors
        assert transform.buffer_state["mean"].dtype == torch.float32
        assert transform.buffer_state["std"].dtype == torch.float32

    def test_invalid_tensor_type(self):
        """Test error for invalid tensor type in get_torch_tensor."""
        features = {
            "state": Feature(
                normalization_data=NormalizationParameters(mean="invalid", std=1),
                shape=(1,),
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: NormalizationType.MEAN_STD}

        with pytest.raises(TypeError, match="list, int, np.ndarray, or torch.Tensor expected"):
            FeatureNormalizeTransform(features, norm_map)

    def test_roundtrip_normalization(self, sample_features, norm_map, sample_batch):
        """Test that forward + inverse normalization recovers original values."""
        forward_transform = FeatureNormalizeTransform(sample_features, norm_map, inverse=False)
        inverse_transform = FeatureNormalizeTransform(sample_features, norm_map, inverse=True)

        original_batch = sample_batch.copy()
        original_state = original_batch["state"].clone()
        original_action = original_batch["action"].clone()

        # Forward normalization
        normalized_batch = forward_transform(original_batch)

        # Inverse normalization
        recovered_batch = inverse_transform(normalized_batch)

        # Should recover original values (within numerical precision)
        torch.testing.assert_close(recovered_batch["state"], original_state, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(recovered_batch["action"], original_action, rtol=1e-4, atol=1e-6)

    def test_empty_shape_handling(self):
        """Test handling of features with empty shape."""
        features = {
            "scalar": Feature(
                normalization_data=NormalizationParameters(mean=0, std=1),
                shape=None,  # Empty shape
                ftype=FeatureType.STATE
            )
        }
        norm_map = {FeatureType.STATE: NormalizationType.MEAN_STD}

        transform = FeatureNormalizeTransform(features, norm_map)

        # Should handle empty shape gracefully
        assert transform.buffer_scalar["mean"].shape == ()
        assert transform.buffer_scalar["std"].shape == ()
